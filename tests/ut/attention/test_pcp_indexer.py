# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.attention.context_parallel.pcp_indexer import (
    PCPIndexerMetadata,
    build_pcp_indexer_batch_metadata,
    build_pcp_indexer_metadata,
    finish_pcp_indexer_topk,
    merge_topk_candidates,
    pack_topk_candidates,
    partition_indexer_keys,
    prepare_pcp_indexer_queries,
    replicate_indexer_query,
    select_longest_decode,
    select_pcp_indexer_candidates,
    supports_pcp_indexer_sharding,
)
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily


@pytest.mark.parametrize(
    "scheduled,computed,prefilling,minimum,expected",
    [
        ([1, 1, 1], [9, 99, 99], [False] * 3, 0, 1),
        ([0, 1, 2, 1], [999, 100, 999, 90], [False, True, False, False], 0, 3),
        ([1], [99], [False], 100, 0),
        ([1], [99], [False], 101, None),
        ([1], [99], [True], 0, None),
        ([], [], [], 0, None),
    ],
)
def test_select_longest_scheduled_decode(scheduled, computed, prefilling, minimum, expected):
    assert select_longest_decode(scheduled, computed, prefilling, minimum) == expected


@pytest.mark.parametrize("length", [1, 127, 128, 129, 4097])
@pytest.mark.parametrize("ranks", [2, 4, 8])
def test_key_partitions_cover_every_visible_position_once(length, ranks):
    positions = []
    sizes = []
    for rank in range(ranks):
        start, count = partition_indexer_keys(length, 128, rank, ranks)
        assert start % 128 == count % 128 == 0
        sizes.append(count)
        positions.extend(range(start, min(start + count, length)))
    assert positions == list(range(length))
    assert max(sizes) - min(sizes) <= 128


@pytest.mark.parametrize("rank,actual_tokens,expected_row", [(0, 4, -1), (1, 4, 1), (1, 0, -1)])
def test_metadata_uses_restore_mapping_and_only_masks_owner(monkeypatch, rank, actual_tokens, expected_row):
    group = SimpleNamespace(rank_in_group=rank, world_size=2)
    monkeypatch.setattr("vllm_ascend.attention.context_parallel.pcp_indexer.get_pcp_group", lambda: group)
    table = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [9, 2, 8, 3], [0, 1, 2, 3]])
    batch = SimpleNamespace(
        is_dummy=False,
        num_reqs=4,
        num_scheduled_tokens=np.array([1, 6, 1, 1]),
        num_computed_tokens_np=np.array([5, 100, 511, 2]),
        is_prefilling_np=np.array([False, True, False, False]),
        query_start_loc_np=np.array([0, 1, 7, 8, 9]),
        seq_lens=torch.tensor([6, 106, 512, 3]),
    )
    context = SimpleNamespace(
        global_batch=batch,
        global_block_tables=(table,),
        hidden_restore_idx=torch.tensor([0, 2, 3, 6, 7, 8, 9, 5, 4]),
    )
    common = SimpleNamespace(
        num_input_tokens=4,
        num_actual_tokens=actual_tokens,
        num_reqs=3,
        query_start_loc=torch.tensor([0, 1, 2, 4]),
        seq_lens=torch.tensor([3, 512, 106]),
        block_table_tensor=table[:3],
    )
    metadata = build_pcp_indexer_metadata(common, context, 0, 128, 128)
    assert metadata is not None
    assert metadata.query_row.tolist() == [expected_row]
    assert (metadata.key_start, metadata.key_count) == (rank * 256, 256)
    torch.testing.assert_close(metadata.block_table, table[2:3])
    assert metadata.seq_len.tolist() == [512]
    assert metadata.batch.key_lens.tolist() == [3, 256 if expected_row == 1 else 512, 106, 256]
    assert common.seq_lens.tolist() == [3, 512, 106]


@pytest.mark.parametrize("owner", [False, True])
def test_native_batch_rebases_only_shared_decode_and_keeps_prefill_causal(owner):
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
        seq_lens=torch.tensor([256, 129], dtype=torch.int32),
        block_table_tensor=torch.tensor([[7, 3, 0], [4, 8, 0]], dtype=torch.int32),
    )
    metadata = PCPIndexerMetadata(
        query_row=torch.tensor([0 if owner else -1]),
        block_table=torch.tensor([[7, 3]], dtype=torch.int32),
        seq_len=torch.tensor([256], dtype=torch.int32),
        key_start=128,
        key_count=128,
    )
    batch = build_pcp_indexer_batch_metadata(common, metadata, 128)
    assert batch.query_lens.tolist() == [1, 4, 4 if owner else 5]
    assert batch.key_lens.tolist() == [128 if owner else 256, 129, 128]
    assert batch.block_table.tolist() == [[3, 0, 0] if owner else [7, 3, 0], [4, 8, 0], [3, 0, 0]]
    assert batch.shared_row.tolist() == [0 if owner else 4]
    assert batch.shard_seq_len.tolist() == [128]
    assert common.seq_lens.tolist() == [256, 129]
    assert common.block_table_tensor.tolist() == [[7, 3, 0], [4, 8, 0]]


@pytest.mark.parametrize("local_reqs", [0, 1])
def test_native_batch_empty_helper_and_empty_shard(local_reqs):
    common = SimpleNamespace(
        num_reqs=local_reqs,
        num_actual_tokens=0,
        query_start_loc=torch.zeros(local_reqs + 1, dtype=torch.int32),
        seq_lens=torch.ones(local_reqs, dtype=torch.int32),
        block_table_tensor=torch.zeros(local_reqs, 1, dtype=torch.int32),
    )
    metadata = PCPIndexerMetadata(
        query_row=torch.tensor([-1]),
        block_table=torch.tensor([[5]], dtype=torch.int32),
        seq_len=torch.tensor([1], dtype=torch.int32),
        key_start=0,
        key_count=0,
    )
    batch = build_pcp_indexer_batch_metadata(common, metadata, 128)
    assert batch.query_lens.tolist() == [0] * local_reqs + [1]
    assert batch.key_lens.tolist() == [1] * (local_reqs + 1)
    assert batch.shared_row.tolist() == [0]
    assert batch.shard_seq_len.tolist() == [0]


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("rows,local_row", [(3, 1), (3, -1), (0, -1)])
def test_replicate_projected_query_includes_scales_and_empty_helpers(quantized, rows, local_row):
    heads, dim = 2, 4
    dtype = torch.int8 if quantized else torch.bfloat16
    shared_q = torch.arange(heads * dim).reshape(1, heads, dim).to(dtype)
    shared_w = torch.tensor([[0.5, -2.0]], dtype=torch.bfloat16)
    shared_s = torch.tensor([[0.25, 0.125]]) if quantized else None
    parts = [shared_q.flatten().float(), shared_w.flatten().float()]
    if shared_s is not None:
        parts.append(shared_s.flatten())
    payload = torch.cat(parts)
    group = SimpleNamespace(all_reduce=Mock(return_value=payload))
    q = torch.zeros(rows, heads, dim, dtype=dtype)
    w = torch.zeros(rows, heads, dtype=shared_w.dtype)
    scale = torch.ones(rows, heads) if quantized else None
    if local_row >= 0:
        q[local_row] = shared_q[0]
        w[local_row] = shared_w[0]
        if scale is not None:
            scale[local_row] = shared_s[0]
    actual_q, actual_w, actual_s = replicate_indexer_query(q, w, scale, torch.tensor([local_row]), group)
    contributed = group.all_reduce.call_args.args[0]
    torch.testing.assert_close(contributed, payload if local_row >= 0 else torch.zeros_like(payload))
    torch.testing.assert_close(actual_q, shared_q)
    torch.testing.assert_close(actual_w, shared_w.float())
    if quantized:
        torch.testing.assert_close(actual_s, shared_s)
    else:
        assert actual_s is None


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("local_row,actual", [(1, 2), (-1, 2), (-1, 0)])
def test_native_query_batch_excludes_padding_and_replicates_once(monkeypatch, quantized, local_row, actual):
    heads, dim = 2, 4
    dtype = torch.int8 if quantized else torch.bfloat16
    q = torch.arange(4 * heads * dim).reshape(4, heads, dim).to(dtype)
    w = torch.arange(4 * heads).reshape(4, heads).to(torch.float16)
    scales = torch.ones(4 * heads, 1, dtype=torch.float16) if quantized else None
    shared_q = q[1:2].clone()
    shared_w = w[1:2].clone()
    parts = [shared_q.flatten().float(), shared_w.flatten().float()]
    if quantized:
        parts.append(torch.ones(heads))
    group = SimpleNamespace(all_reduce=Mock(return_value=torch.cat(parts)))
    monkeypatch.setattr("vllm_ascend.attention.context_parallel.pcp_indexer.get_pcp_group", lambda: group)
    metadata = SimpleNamespace(query_row=torch.tensor([local_row]), batch=SimpleNamespace(num_local_tokens=actual))
    combined_q, combined_w, combined_s = prepare_pcp_indexer_queries(q, w, scales, metadata)
    assert group.all_reduce.call_count == 1
    torch.testing.assert_close(combined_q, torch.cat((q[:actual], shared_q)))
    torch.testing.assert_close(combined_w, torch.cat((w[:actual], shared_w)))
    if quantized:
        assert combined_s.shape == (actual + 1, heads)
        assert combined_s.dtype == scales.dtype
    else:
        assert combined_s is None


@pytest.mark.parametrize(
    "scores",
    [
        [100, 99, 98, 97, 1, 0, -1, -2, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [-9, -3, -7, -2, -1, -6, -8, -5, -4, -10, -11, -12],
    ],
)
def test_local_candidates_merge_to_full_topk_including_ties(scores):
    scores = torch.tensor(scores, dtype=torch.float32)
    candidates = torch.cat(
        [
            pack_topk_candidates(torch.arange(4), scores[start : start + 4], start, torch.tensor([4]))
            for start in (0, 4, 8)
        ]
    )
    result = merge_topk_candidates(candidates, 4).flatten()
    expected = torch.argsort(scores, descending=True, stable=True)[:4].to(torch.int32)
    torch.testing.assert_close(result, expected)


def test_padding_and_large_token_ids_survive_candidate_exchange():
    start = 2**24 + 1
    candidates = pack_topk_candidates(
        torch.arange(4), torch.tensor([1.0, 2.0, -float("inf"), -float("inf")]), start, torch.tensor([4])
    )
    empty = pack_topk_candidates(torch.arange(4), torch.full((4,), -float("inf")), start + 4, torch.tensor([0]))
    merged = merge_topk_candidates(torch.cat((candidates, empty)), 4)
    assert merged.flatten().tolist() == [start + 1, start, -1, -1]


@pytest.mark.parametrize("local_row,actual", [(1, 2), (-1, 2), (-1, 0)])
def test_finish_uses_native_scores_and_preserves_other_rows(monkeypatch, local_row, actual):
    native = torch.tensor([[[0, 1, -1, -1]], [[3, 1, 2, 0]], [[3, 1, 2, 0]]], dtype=torch.int32)
    shared_row = local_row if local_row >= 0 else actual
    if actual == 0:
        native = native[2:]
    batch = SimpleNamespace(
        shared_row=torch.tensor([shared_row]),
        shard_seq_len=torch.tensor([4]),
        num_local_tokens=actual,
    )
    metadata = SimpleNamespace(query_row=torch.tensor([local_row]), key_start=128, batch=batch)
    scores = torch.tensor([3.0, 1.0, 2.0, 0.0]).expand_as(native).contiguous()
    other = pack_topk_candidates(torch.arange(4), torch.tensor([5.0, 4.0, -1.0, -2.0]), 256, torch.tensor([4]))
    group = SimpleNamespace(all_gather=Mock(side_effect=lambda candidates, dim: torch.cat((candidates, other))))
    monkeypatch.setattr("vllm_ascend.attention.context_parallel.pcp_indexer.get_pcp_group", lambda: group)
    result = finish_pcp_indexer_topk(native, scores, metadata, 4)
    assert group.all_gather.call_count == 1
    expected = torch.full((4, 1, 4), -1, dtype=torch.int32)
    expected[:actual] = native[:actual]
    if local_row >= 0:
        expected[local_row, 0] = torch.tensor([256, 257, 131, 130])
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16, torch.float16, torch.float8_e4m3fn])
def test_native_sharding_preserves_non_int8_layers(monkeypatch, dtype):
    for family in DeviceAdaptorFamily:
        monkeypatch.setattr(
            "vllm_ascend.attention.context_parallel.pcp_indexer.get_current_hardware_profile",
            lambda family=family: SimpleNamespace(device_adaptor_family=family),
        )
        assert supports_pcp_indexer_sharding(dtype, 32, 128, 2048) == (
            family == DeviceAdaptorFamily.STANDARD and dtype == torch.int8
        )
        assert not supports_pcp_indexer_sharding(dtype, 8, 128, 2048)
        assert not supports_pcp_indexer_sharding(dtype, 32, 64, 2048)
        assert not supports_pcp_indexer_sharding(dtype, 32, 128, 4096)


def test_v2_returns_scores_once_and_reuses_batch_metadata_across_layers(monkeypatch):
    q = torch.zeros(3, 32, 128, dtype=torch.int8)
    cache = torch.zeros(8, 128, 1, 128, dtype=torch.int8)[::2]
    scales = torch.ones(8, 128, 1, 1, dtype=torch.float16)[::2]
    batch = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 2], dtype=torch.int32),
        key_lens=torch.tensor([129, 128], dtype=torch.int32),
        block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        native_metadata={},
    )
    produced = torch.zeros(1024, dtype=torch.int32)
    outputs = (torch.zeros(3, 1, 2048, dtype=torch.int32), torch.ones(3, 1, 2048))
    producer, native = Mock(return_value=produced), Mock(return_value=outputs)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata", producer, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2", native, raising=False)
    for _ in range(2):
        assert (
            select_pcp_indexer_candidates(
                q, torch.ones(3, 32), torch.ones(3, 32), cache, scales, SimpleNamespace(batch=batch), 2048
            )
            is outputs
        )
    assert producer.call_count == 1
    assert native.call_count == 2
    kwargs = native.call_args.kwargs
    assert kwargs["return_value"] == 1 and kwargs["enable_dcp"] is False
    assert kwargs["mask_mode"] == 3 and kwargs["cmp_ratio"] == 1
    assert kwargs["layout_k"] == "PA_BBND"
    assert kwargs["metadata"] is produced
    assert kwargs["cu_seqlens_q"] is batch.query_start_loc
    assert kwargs["seqused_k"] is batch.key_lens
    assert kwargs["block_table"] is batch.block_table
    assert native.call_args.args[1] is cache
    assert native.call_args.args[4].stride(0) == scales.stride(0)
    assert all(t.dtype == torch.float16 for t in native.call_args.args[2:5])
