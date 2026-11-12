# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cooperative indexing of one decode query without changing PCP ownership."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.distributed import get_pcp_group

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile


@dataclass
class PCPIndexerBatchMetadata:
    """One native indexer batch containing local queries and the shared shard."""

    block_table: torch.Tensor
    query_lens: torch.Tensor
    query_start_loc: torch.Tensor
    key_lens: torch.Tensor
    shared_row: torch.Tensor
    shard_seq_len: torch.Tensor
    num_local_tokens: int
    # The producer depends on batch boundaries and shape, not layer Q/K values.
    native_metadata: dict[tuple[int, int, int], torch.Tensor] = field(default_factory=dict)


@dataclass
class PCPIndexerMetadata:
    # -1 on helper ranks; otherwise the query's row in the owner's local batch.
    query_row: torch.Tensor
    block_table: torch.Tensor
    seq_len: torch.Tensor
    key_start: int
    key_count: int
    batch: PCPIndexerBatchMetadata | None = None


def select_longest_decode(
    scheduled: Sequence[int],
    computed: Sequence[int],
    is_prefilling: Sequence[bool],
    min_seq_len: int,
) -> int | None:
    """Choose a single-token decode, breaking length ties by global batch order."""
    candidates = (
        i for i, count in enumerate(scheduled) if count == 1 and not is_prefilling[i] and computed[i] + 1 >= min_seq_len
    )
    return max(candidates, key=lambda i: (computed[i], -i), default=None)


def partition_indexer_keys(seq_len: int, block_size: int, rank: int, world_size: int) -> tuple[int, int]:
    """Partition logical blocks, including the partially filled last block."""
    blocks = (seq_len + block_size - 1) // block_size
    start = blocks * rank // world_size * block_size
    end = blocks * (rank + 1) // world_size * block_size
    return start, end - start


def build_pcp_indexer_metadata(
    common: Any,
    context: Any,
    cache_group: int,
    block_size: int,
    min_seq_len: int,
) -> PCPIndexerMetadata | None:
    batch = context.global_batch
    if batch.is_dummy:
        return None
    count = batch.num_reqs
    request = select_longest_decode(
        batch.num_scheduled_tokens[:count],
        batch.num_computed_tokens_np[:count],
        batch.is_prefilling_np[:count],
        min_seq_len,
    )
    if request is None:
        return None

    group = get_pcp_group()
    global_token = int(batch.query_start_loc_np[request])
    # Use the manager's actual restore mapping, rather than duplicating its
    # ownership/reordering policy. No device-to-host scalar read is needed.
    gathered_row = context.hidden_restore_idx[global_token : global_token + 1]
    local_start = group.rank_in_group * common.num_input_tokens
    local_row = gathered_row - local_start
    owns_query = (local_row >= 0) & (local_row < common.num_actual_tokens)
    local_row = torch.where(owns_query, local_row, -1)
    seq_len = int(batch.num_computed_tokens_np[request]) + 1
    key_start, key_count = partition_indexer_keys(seq_len, block_size, group.rank_in_group, group.world_size)
    block_count = (seq_len + block_size - 1) // block_size
    metadata = PCPIndexerMetadata(
        query_row=local_row,
        block_table=context.global_block_tables[cache_group][request : request + 1, :block_count],
        seq_len=batch.seq_lens[request : request + 1],
        key_start=key_start,
        key_count=key_count,
    )
    metadata.batch = build_pcp_indexer_batch_metadata(common, metadata, block_size)
    return metadata


def build_pcp_indexer_batch_metadata(
    common: Any, metadata: PCPIndexerMetadata, block_size: int
) -> PCPIndexerBatchMetadata:
    """Rebase only the selected decode's K view, preserving local causal rows.

    The owner uses its existing query row. Helpers append one query. An extra
    storage row on the owner has zero query length, so A is never indexed twice.
    All ownership decisions stay on device, including empty/padded local ranks.
    """
    count = common.num_reqs
    local_tokens = common.num_actual_tokens
    query_lens = common.query_start_loc[1 : count + 1].clamp(max=local_tokens)
    starts = common.query_start_loc[:count]
    owner = metadata.query_row >= 0
    selected = owner & (starts <= metadata.query_row) & (metadata.query_row < query_lens)

    block_start = metadata.key_start // block_size
    block_count = metadata.key_count // block_size
    shard_table = metadata.block_table[:, block_start : block_start + block_count]
    local_table = common.block_table_tensor[:count]
    width = max(local_table.shape[1], shard_table.shape[1], 1)
    shard_table = torch.nn.functional.pad(shard_table, (0, width - shard_table.shape[1]))
    local_table = torch.nn.functional.pad(local_table, (0, width - local_table.shape[1]))
    block_table = torch.cat((torch.where(selected[:, None], shard_table, local_table), shard_table), dim=0)

    shard_len = (metadata.seq_len - metadata.key_start).clamp(min=0, max=metadata.key_count)
    # Some CANN kernels reject zero K length. Empty shards scan one placeholder
    # key; the real zero length is retained to invalidate all their candidates.
    native_shard_len = shard_len.clamp(min=1)
    key_lens = torch.cat((torch.where(selected, native_shard_len, common.seq_lens[:count]), native_shard_len))
    appended_end = torch.full_like(metadata.query_row, local_tokens) + (~owner).to(metadata.query_row.dtype)
    query_lens = torch.cat((query_lens, appended_end.to(query_lens.dtype)))
    shared_row = torch.where(owner, metadata.query_row, local_tokens)
    return PCPIndexerBatchMetadata(
        block_table=block_table,
        query_lens=query_lens,
        query_start_loc=torch.nn.functional.pad(query_lens, (1, 0)),
        key_lens=key_lens,
        shared_row=shared_row,
        shard_seq_len=shard_len,
        num_local_tokens=local_tokens,
    )


def supports_pcp_indexer_sharding(dtype: torch.dtype, heads: int, head_dim: int, topk: int) -> bool:
    """PR #20878's arch22 score output accepts INT8 Q/K and FP16 scales.

    Preserve floating-point fallback layers instead of quantizing their cache
    merely to enter the cooperative path.
    """
    return (
        get_current_hardware_profile().device_adaptor_family == DeviceAdaptorFamily.STANDARD
        and dtype == torch.int8
        and heads in (32, 64)
        and head_dim == 128
        and 0 < topk <= 2048
    )


def select_pcp_indexer_candidates(
    query: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_cache: torch.Tensor,
    key_scale: torch.Tensor,
    metadata: PCPIndexerMetadata,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One PR #20878 QLI V2 call returns both local indices and FP32 scores.

    PA_BBND is V2's spelling for the existing [blocks, block_size, N, D]
    cache. PCP partitions only A's table/length; operator DCP stays disabled.
    Preserve the native score scale (arch22 uses QK / 1024) on every rank.
    """
    assert metadata.batch is not None
    batch = metadata.batch
    heads, dim = query.shape[1:]
    shape_key = (heads, dim, topk)
    if shape_key not in batch.native_metadata:
        batch.native_metadata[shape_key] = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            heads,
            1,
            dim,
            topk,
            2,  # V2 QUANT_MODE_INT8 (the old torch_npu API used 0).
            cu_seqlens_q=batch.query_start_loc,
            seqused_k=batch.key_lens,
            batch_size=batch.key_lens.shape[0],
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
        )
    return torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query.contiguous(),
        key_cache,
        weights.to(torch.float16).contiguous(),
        query_scale.reshape(query.shape[:-1]).to(torch.float16).contiguous(),
        key_scale.reshape(key_cache.shape[:2] + (1,)),
        topk,
        2,
        cu_seqlens_q=batch.query_start_loc,
        seqused_k=batch.key_lens,
        block_table=batch.block_table,
        metadata=batch.native_metadata[shape_key],
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
        return_value=1,
        enable_dcp=False,
    )


def replicate_indexer_query(
    query: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor | None,
    query_row: torch.Tensor,
    group: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Replicate the owner's projected query in one collective, including empty ranks."""
    heads, dim = query.shape[1:]
    query_size = heads * dim
    scale_size = heads if query_scale is not None else 0
    payload = torch.zeros(query_size + heads + scale_size, dtype=torch.float32, device=query.device)
    if query.shape[0]:
        row = query_row.clamp(min=0)
        owned = query_row >= 0
        payload[:query_size] = torch.where(owned, query.index_select(0, row).flatten().float(), 0)
        payload[query_size : query_size + heads] = torch.where(owned, weights.index_select(0, row).flatten().float(), 0)
        if query_scale is not None:
            payload[query_size + heads :] = torch.where(
                owned, query_scale.reshape(-1, heads).index_select(0, row).flatten().float(), 0
            )
    # Exactly one rank contributes. FP32 preserves BF16/FP16/INT8/FP8 query
    # values and their scales, and avoids a data-dependent broadcast root.
    payload = group.all_reduce(payload)
    shared_query = payload[:query_size].to(query.dtype).view(1, heads, dim)
    shared_weights = payload[query_size : query_size + heads].view(1, heads)
    shared_scale = payload[query_size + heads :].view(1, heads) if query_scale is not None else None
    return shared_query, shared_weights, shared_scale


def prepare_pcp_indexer_queries(
    query: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor | None,
    metadata: PCPIndexerMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Replicate the selected Q and append it after actual local tokens.

    The owner retains its original Q row; its appended storage row has zero
    query length in ``metadata.batch``. Helpers use the appended row. Excluding
    input padding keeps TND query boundaries valid on partially empty ranks.
    """
    assert metadata.batch is not None
    shared_q, shared_w, shared_s = replicate_indexer_query(
        query, weights, query_scale, metadata.query_row, get_pcp_group()
    )
    count = metadata.batch.num_local_tokens
    batched_q = torch.cat((query[:count], shared_q))
    batched_w = torch.cat((weights[:count], shared_w.to(weights.dtype)))
    batched_s = None
    if query_scale is not None:
        assert shared_s is not None
        scales = query_scale.reshape(-1, query.shape[1])
        batched_s = torch.cat((scales[:count], shared_s.to(query_scale.dtype)))
    return batched_q, batched_w, batched_s


def pack_topk_candidates(
    indices: torch.Tensor, scores: torch.Tensor, key_start: int, seq_len: torch.Tensor
) -> torch.Tensor:
    """Pack native candidates with FP32 scores and global int32 token IDs."""
    indices = indices.flatten().to(torch.int32)
    valid = (indices >= 0) & (indices < seq_len) & ~torch.isnan(scores) & (scores > -float("inf"))
    values = scores.masked_fill(~valid, -float("inf"))
    indices = torch.where(valid, indices + key_start, -1)
    # Native outputs need not be ordered, especially when K length <= top-k.
    # Order just the K candidate IDs so the stable merge breaks score ties by ID.
    indices, order = torch.sort(indices, stable=True)
    values = values.index_select(0, order).contiguous()
    return torch.stack((values.view(torch.int32), indices), dim=-1)


def merge_topk_candidates(candidates: torch.Tensor, topk: int) -> torch.Tensor:
    """Merge rank-major candidates; equal scores prefer the lower global token ID."""
    scores = candidates[:, 0].contiguous().view(torch.float32)
    _, order = torch.sort(scores, descending=True, stable=True)
    return candidates[:, 1].index_select(0, order[:topk]).view(1, 1, topk)


def finish_pcp_indexer_topk(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    metadata: PCPIndexerMetadata,
    output_tokens: int,
) -> torch.Tensor:
    """Merge A's native candidates and restore the original local output rows.

    Scores come directly from the native selection. No Q/K scoring is done here.
    """
    assert metadata.batch is not None
    batch = metadata.batch
    group = get_pcp_group()
    indices = topk_indices.index_select(0, batch.shared_row).flatten()
    scores = topk_scores.index_select(0, batch.shared_row).flatten().float()
    candidates = pack_topk_candidates(indices, scores, metadata.key_start, batch.shard_seq_len)
    gathered = group.all_gather(candidates, dim=0)
    shared_topk = merge_topk_candidates(gathered, topk_indices.shape[-1])
    local_topk = torch.nn.functional.pad(
        topk_indices[: batch.num_local_tokens], (0, 0, 0, 0, 0, output_tokens - batch.num_local_tokens), value=-1
    )
    selected = torch.arange(output_tokens, device=local_topk.device) == metadata.query_row
    return torch.where(selected.view(-1, 1, 1), shared_topk.to(local_topk.dtype), local_topk)
