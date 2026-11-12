# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One native indexer call per rank; run with torchrun and pytest --noconftest."""

import os
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

from vllm_ascend.attention.context_parallel import pcp_indexer

torch_npu = pytest.importorskip("torch_npu")
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(WORLD_SIZE not in (2, 4), reason="requires PCP2 or PCP4")


def native_topk(q, cache, weights, table, query_lens, key_lens, topk, qscale, kscale):
    batch = SimpleNamespace(
        query_start_loc=torch.nn.functional.pad(query_lens, (1, 0)),
        key_lens=key_lens,
        block_table=table.contiguous(),
        native_metadata={},
    )
    return pcp_indexer.select_pcp_indexer_candidates(
        q, weights, qscale, cache, kscale, SimpleNamespace(batch=batch), topk
    )


@torch.inference_mode()
def test_single_native_batch_with_changing_owner_mixed_prefill_and_empty_helpers(monkeypatch):
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    init_distributed_environment(
        world_size=WORLD_SIZE, rank=rank, local_rank=local_rank, distributed_init_method="env://", backend="hccl"
    )
    group = init_model_parallel_group(
        [list(range(WORLD_SIZE))],
        local_rank=local_rank,
        backend="hccl",
        group_name="pcp_indexer_test",
        use_device_communicator=True,
    )
    monkeypatch.setattr(pcp_indexer, "get_pcp_group", lambda: group)
    try:
        heads, dim = 32, 128
        for dtype in (torch.int8,):
            for length, topk in ((1, 32), (257, 2048), (8193, 2048)):
                blocks = (length + 127) // 128
                torch.manual_seed(14)
                cache_cpu = torch.randint(-4, 5, (blocks + 2, 128, 1, dim)).to(dtype)
                cache = cache_cpu.npu()
                table = torch.arange(blocks, dtype=torch.int32, device="npu").flip(0).view(1, -1)
                width = max(blocks, 2)
                full_table = torch.nn.functional.pad(table, (0, width - blocks))
                other_table = torch.zeros(1, width, dtype=torch.int32, device="npu")
                other_table[0, :2] = torch.tensor([blocks, blocks + 1], device="npu")
                seq_len = torch.tensor([length], dtype=torch.int32, device="npu")
                q_all = torch.randint(-3, 4, (4, heads, dim)).to(dtype).npu()
                wdtype = torch.float16 if dtype == torch.int8 else dtype
                weights_all = torch.arange(heads, device="npu").sub(heads // 2).expand(4, -1).to(wdtype).contiguous()
                scales = torch.ones(4, heads, dtype=torch.float16, device="npu") if dtype == torch.int8 else None
                kscale = (
                    torch.ones(blocks + 2, 128, 1, dtype=torch.float16, device="npu") if scales is not None else None
                )
                other_baseline, _ = native_topk(
                    q_all[1:],
                    cache,
                    weights_all[1:],
                    other_table,
                    torch.tensor([3], dtype=torch.int32, device="npu"),
                    torch.tensor([129], dtype=torch.int32, device="npu"),
                    topk,
                    scales[1:] if scales is not None else None,
                    kscale,
                )
                logical_keys = cache_cpu[:blocks].flip(0).reshape(-1, dim)[:length].float()
                reference = (
                    (q_all[0].cpu().float() @ logical_keys.T).clamp_min(0) * weights_all[0].cpu().float().view(-1, 1)
                ).sum(0)
                for owner in (0, WORLD_SIZE - 1):
                    for with_local_prefill in (False, True):
                        owns = rank == owner
                        # A is one decode token; B is a three-token prefill chunk.
                        parts = ([q_all[:1]] if owns else []) + ([q_all[1:]] if with_local_prefill else [])
                        wparts = ([weights_all[:1]] if owns else []) + ([weights_all[1:]] if with_local_prefill else [])
                        local = torch.cat(parts) if parts else q_all[:0]
                        local_w = torch.cat(wparts) if wparts else weights_all[:0]
                        count = local.shape[0]
                        # Model-input padding must never become an extra request.
                        q = torch.cat((local, torch.zeros_like(q_all[:2])))
                        w = torch.cat((local_w, torch.zeros_like(weights_all[:2])))
                        qscale = (
                            torch.ones(count + 2, heads, dtype=torch.float16, device="npu")
                            if scales is not None
                            else None
                        )
                        tables = ([full_table] if owns else []) + ([other_table] if with_local_prefill else [])
                        local_table = (
                            torch.cat(tables) if tables else torch.empty(0, width, dtype=torch.int32, device="npu")
                        )
                        ends = ([1] if owns else []) + ([count] if with_local_prefill else [])
                        lengths = ([length] if owns else []) + ([129] if with_local_prefill else [])
                        common = SimpleNamespace(
                            num_reqs=len(ends),
                            num_actual_tokens=count,
                            query_start_loc=torch.tensor([0] + ends, dtype=torch.int32, device="npu"),
                            seq_lens=torch.tensor(lengths, dtype=torch.int32, device="npu"),
                            block_table_tensor=local_table,
                        )
                        start, size = pcp_indexer.partition_indexer_keys(length, 128, rank, WORLD_SIZE)
                        metadata = pcp_indexer.PCPIndexerMetadata(
                            query_row=torch.tensor([0 if owns else -1], device="npu"),
                            block_table=table,
                            seq_len=seq_len,
                            key_start=start,
                            key_count=size,
                        )
                        batch = metadata.batch = pcp_indexer.build_pcp_indexer_batch_metadata(common, metadata, 128)
                        combined_q, combined_w, combined_scale = pcp_indexer.prepare_pcp_indexer_queries(
                            q, w, qscale, metadata
                        )
                        # Exactly one native selection for all local queries + A.
                        native, native_scores = native_topk(
                            combined_q,
                            cache,
                            combined_w,
                            batch.block_table,
                            batch.query_lens,
                            batch.key_lens,
                            topk,
                            combined_scale,
                            kscale,
                        )
                        result = pcp_indexer.finish_pcp_indexer_topk(native, native_scores, metadata, q.shape[0]).cpu()
                        assert (result[count:] == -1).all()
                        if with_local_prefill:
                            torch.testing.assert_close(result[int(owns) : count], other_baseline.cpu())
                        if owns:
                            selected = result[0].flatten()
                            valid = selected[selected >= 0].long()
                            assert valid.numel() == min(length, topk)
                            assert valid.unique().numel() == valid.numel()
                            # Native tie-breaking may differ at the cut; every
                            # selected score must meet the full-context cutoff.
                            threshold = reference.topk(min(length, topk)).values[-1]
                            assert (reference[valid] >= threshold).all()
                            assert (valid < length).all()
    finally:
        group.destroy()
        destroy_distributed_environment()
