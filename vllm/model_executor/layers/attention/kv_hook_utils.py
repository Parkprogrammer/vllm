"""KV Cache Hook Utilities for Post-hoc Attention Analysis

This module provides utilities to capture and analyze attention patterns
after request completion with zero generation overhead.
"""

from dataclasses import dataclass
from bisect import bisect_left
from typing import Dict, List, Optional, Set, Tuple, Any

import logging
import re, time, struct, pickle
import gzip, base64
import torch
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)

def _shm_name(req_id: str) -> str:
    """Deterministic shared-memory segment name from request ID."""
    return "/vkv_" + req_id.replace("-", "")[:40]

def _shm_write(req_id: str, snapshots: list[dict]) -> None:
    """Write snapshot list to a named shared-memory segment.

    Protocol: size header is written LAST so readers treat size==0
    as "write in progress" and keep polling.
    """
    payload = pickle.dumps(snapshots)  # Trust boundary: only vLLM worker writes
    size, name = len(payload), _shm_name(req_id)
    # Clean up stale segment from a previous failed run
    try:
        stale = shared_memory.SharedMemory(name=name, create=False)
        stale.close()
        stale.unlink()
    except FileNotFoundError:
        pass
    mem = shared_memory.SharedMemory(name=name, create=True, size=8 + size)
    mem.buf[8:8 + size] = payload          # data first
    struct.pack_into("Q", mem.buf, 0, size)  # size LAST (ready signal)
    mem.close()


def _shm_read(req_id: str, timeout: float = 5.0) -> list[dict] | None:
    """Read snapshot list from shared-memory, polling until available.

    Handles race condition: if the segment exists but size==0 the writer
    hasn't finished yet — close and retry.  Also catches corrupt reads
    from partially-written segments.
    """
    name, deadline = _shm_name(req_id), time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            mem = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        size = struct.unpack_from("Q", mem.buf, 0)[0]
        if size == 0:
            # Writer created segment but hasn't finished writing yet
            mem.close()
            time.sleep(0.01)
            continue
        try:
            data = pickle.loads(bytes(mem.buf[8:8 + size]))
        except Exception:
            # Corrupt read — writer may still be flushing; retry
            mem.close()
            time.sleep(0.01)
            continue
        mem.close()
        mem.unlink()
        return data
    # Timeout: clean up orphaned segment to prevent leaks
    try:
        mem = shared_memory.SharedMemory(name=name, create=False)
        mem.close()
        mem.unlink()
    except FileNotFoundError:
        pass
    return None


def extract_k_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_ids: list[int],
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Extract K vectors from paged KV cache at given slot positions.

    Backend-agnostic: auto-detects layout from tensor shape.
      - FlashInfer:    [num_blocks, 2, page_size, num_kv_heads, head_dim]
      - FlashAttention:[2, num_blocks, page_size, num_kv_heads, head_dim]

    Returns: Tensor of shape [len(slot_ids), num_kv_heads, head_dim]
    """
    shape, slot_tensor = kv_cache.shape, torch.tensor(slot_ids, dtype=torch.long, device=kv_cache.device)

    if kv_cache.ndim == 5 and shape[0] == 2:
        # FlashAttention layout: [2(K/V), num_blocks, page_size, ...]
        page_size, num_slots = shape[2], shape[1] * shape[2]
        k_flat = kv_cache[0].reshape(num_slots, -1)          # [total_slots, kv_heads*head_dim]
        k = k_flat[slot_tensor].view(len(slot_ids), shape[3], shape[4])
    elif kv_cache.ndim == 5 and shape[1] == 2:
        # FlashInfer layout: [num_blocks, 2(K/V), page_size, ...]
        page_size = shape[2]
        page_indices, page_offsets = slot_tensor // page_size, slot_tensor % page_size
        k = kv_cache[page_indices, 0, page_offsets]           # [T, kv_heads, head_dim]
    elif kv_cache.ndim == 3 and shape[0] == 2:
        # Simple layout: [2, total_slots, hidden]
        k = kv_cache[0, slot_tensor]
    else:
        raise ValueError(
            f"Unsupported KV cache layout: ndim={kv_cache.ndim} shape={list(shape)}" )

    return k.to(dtype)  # stay on GPU for fast attention computation


_LAYER_PATTERNS = [
    re.compile(r"(?:^|\.)(?:layers)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:h)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:blocks)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:decoder\.layers)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:model\.layers)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)(?:transformer\.h)\.(\d+)(?:\.|$)"), ]

_kv_hook: Optional['KVHook'] = None

def set_kv_hook(hook: Optional['KVHook']) -> None:
    """Set the global KV hook instance"""
    global _kv_hook
    _kv_hook = hook

def get_kv_hook() -> Optional['KVHook']:
    """Get the global KV hook instance"""
    return _kv_hook

def load_kv_snapshot_data(req_id: str) -> list[dict[str, Any]] | None:
    """Load KV snapshot(s) via shared memory (cross-process, no disk I/O)."""
    return _shm_read(req_id, timeout=5.0)

@dataclass
class HookConfig:
    """Configuration for KV Cache hook"""
    enabled: bool = False
    layers: Set[int] = None

class KVHook:
    """Per-worker KV hook for post-hoc attn_score and other utilities.
    
    This class is instantiated once per ModelRunner and bound to
    each Attention layer via bind_capture_state().
    
    """
    
    def __init__(self, config: HookConfig, model_config=None):
        self.config = config
        self.model_config = model_config
        # NOTE(jehyun) : This buffer is handled per-worker for capturing
        # attention scores at the intended token output for many worker processes
        # (layer_idx, slot_id) -> list of [num_heads, head_dim] tensors
        self.q_buffer: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        # Search the layer name once, keep for the same worker.
        self._layer_idx_cache: Dict[str, int] = {}

        # For tracing snapshot, buffering logic on/off for each batch step
        self.runtime_enabled_this_step = False
        # Per-step set of slot IDs belonging to capture-enabled requests.
        # Only these slots are buffered in buffer_query() to avoid wasting
        # memory on non-capture requests and to prevent stale cross-request
        # data when blocks are reallocated.
        self.capture_slots: Optional[Set[int]] = None

    def buffer_query(self, query: torch.Tensor, key: torch.Tensor, attn_metadata, layer_name: str) -> None:
        """Buffer Query tokens at attention-computation time.

        K is NOT buffered here — it is read directly from the KV cache
        at request completion time via extract_k_from_kv_cache().
        """

        if (not self.config.enabled) or (attn_metadata is None): return

        layer_idx = self._extract_layer_idx(layer_name)
        if layer_idx < 0: return

        slot_ids = attn_metadata.slot_mapping
        if query.shape[0] != slot_ids.shape[0]: return

        try:
            query_cpu = query.detach().cpu().clone()
        except Exception:
            return

        capture_slots = self.capture_slots
        for i in range(query.shape[0]):

            slot_id = slot_ids[i].item()
            if slot_id < 0: continue
            # Only buffer slots belonging to capture-enabled requests
            if capture_slots is not None and slot_id not in capture_slots: continue

            buffer_key = (layer_idx, slot_id)
            if buffer_key not in self.q_buffer: self.q_buffer[buffer_key] = []

            q_token = query_cpu[i].to(torch.float16) if query_cpu[i].dtype != torch.float16 else query_cpu[i]
            self.q_buffer[buffer_key].append(q_token)
    
    def build_token_meta(
        self,
        req_state,
        token_idx: list[int],
        *,
        ordered_slots_len: int | None = None,
    ) -> dict[str, Any]:
        """Build token mapping metadata for post-hoc client-side alignment."""

        prompt_len = int(getattr(req_state, "num_prompt_tokens", 0) or 0)
        total_len = int(getattr(req_state, "num_tokens", prompt_len) or prompt_len)
        # Not sure. Why is there a num_prompt and num_ at the same time?
        if prompt_len > total_len: prompt_len = total_len

        raw_vision_ranges: list[dict[str, int]] = []
        mm_features = getattr(req_state, "mm_features", None) or []
        
        for feature in mm_features:
            
            # Multi-Modal encoder/decoder seperation placeholder
            pos = getattr(feature, "mm_position", None)
            if pos is None: continue
            start, length = int(getattr(pos, "offset", 0) or 0), int(getattr(pos, "length", 0) or 0)
            end = start + max(length, 0)

            # Vision placeholders are in prompt span.
            if end <= 0 or start >= prompt_len: continue
            start, end = max(0, start), min(prompt_len, end)
            if start < end: raw_vision_ranges.append({"start": start, "end": end})

        # Sort and merge overlapping vision ranges.
        raw_vision_ranges.sort(key=lambda r: (r["start"], r["end"]))
        vision_ranges: list[dict[str, int]] = []
        for r in raw_vision_ranges:
            if not vision_ranges or r["start"] > vision_ranges[-1]["end"]:
                vision_ranges.append(dict(r))
            else:
                vision_ranges[-1]["end"] = max(vision_ranges[-1]["end"], r["end"])

        # Complement of vision ranges within prompt range.
        language_ranges: list[dict[str, int]] = []
        cursor = 0
        for r in vision_ranges:
            if cursor < r["start"]:
                language_ranges.append({"start": cursor, "end": r["start"]})
            cursor = max(cursor, r["end"])
            
        # Final target. Image-Text seperation
        if cursor < prompt_len: language_ranges.append({"start": cursor, "end": prompt_len})

        ordered_len = int(ordered_slots_len if ordered_slots_len is not None else len(token_idx))
        window_offset = int(total_len - ordered_len)

        prompt_boundary_local = bisect_left(token_idx, prompt_len) if token_idx else None
        token_idx_shifted = [int(i) + window_offset for i in token_idx]
        prompt_boundary_with_offset = (bisect_left(token_idx_shifted, prompt_len) if token_idx_shifted else None)

        return {
            "token_idx": [int(i) for i in token_idx],
            "prompt_len": prompt_len, 
            "total_len": total_len,
            "vision_ranges": vision_ranges,
            "language_ranges": language_ranges,
            "token_idx_basis": "window_local",
            "window_offset_candidate": window_offset,
            "prompt_boundary_local": prompt_boundary_local,
            "prompt_boundary_with_offset_candidate": prompt_boundary_with_offset, }

    def _compute_attention(self, q_tensor, k_tensor, scale):
        
        # [hq, T, d] | [hk, T, d]
        q, k = q_tensor.transpose(0, 1), k_tensor.transpose(0, 1) 
        hq, Tq, d = q.shape
        hk, Tk, dk = k.shape

        if d != dk or Tq != Tk: return None

        # Always create a mapping index from hk to hq
        if hk == hq:
            k_m = k
        elif hk < hq and (hq % hk == 0):
            # GQA: expand K heads to match Q heads
            k_m = k.index_select(0, torch.arange(hq, device=k.device) // (hq // hk))
        else:
            # Fallback: linear resample
            idx = torch.clamp( 
                torch.floor(torch.arange(hq, device=k.device) * (hk / hq)).long(),
                0, hk - 1)
            k_m = k.index_select(0, idx)

        scores = torch.bmm(q, k_m.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        return probs.transpose(0, 1)  # [hq, T, T] -> [T, hq, T]

    
    @staticmethod
    def _ordered_slots_for_group(
        block_list: list[int], num_tokens: int, block_size: int,
    ) -> list[int]:
        """Build deduplicated, token-order slot list from a block group."""
        ordered: list[int] = []
        seen: Set[int] = set()
        cursor = 0
        for block_id in block_list:
            if cursor >= num_tokens: break
            base = block_id * block_size
            count = min(block_size, num_tokens - cursor)
            for off in range(count):
                slot_id = base + off
                if slot_id not in seen:
                    ordered.append(slot_id); seen.add(slot_id)
            cursor += count
        return ordered

    @staticmethod
    def _slots_from_blocks(block_list: list[int], block_size: int) -> Set[int]:
        """Return the set of all slot IDs covered by the given blocks."""
        slots: Set[int] = set()
        for bid in block_list:
            base = bid * block_size
            for off in range(block_size):
                slots.add(base + off)
        return slots

    @staticmethod
    def _resolve_target_layers(req_state, default_layers: Set[int] | None) -> Set[int]:
        """Determine target layers: per-request override or server default."""
        target = default_layers or set()
        if req_state.sampling_params and req_state.sampling_params.extra_args:
            layers_str = req_state.sampling_params.extra_args.get('kv_hook_layers')
            if layers_str and layers_str.strip().lower() != 'all':
                target = set(int(x.strip()) for x in layers_str.split(','))
        return target

    def capture_kv_q_attention(self, req_state, block_size: int, kv_caches, prefix: str | None = None) -> None:
        """
         At the timing of freeing request of vllm-engine,
         create 1 snapshot of attention scores for the requested req_id
        """
        req_id = None
        try:
            req_id = req_state.req_id
            layers = self._resolve_target_layers(req_state, self.config.layers)
            snapshots: list[dict] = []

            for layer_idx in layers:
                buf_slots = [sid for (li, sid) in self.q_buffer if li == layer_idx]
                if (not buf_slots) or (not req_state.block_ids): continue

                # Find which block_ids group overlaps with buffered slots
                buf_set, grp_idx = set(buf_slots), None
                for gi, block_list in enumerate(req_state.block_ids):
                    if not block_list: continue
                    if buf_set & self._slots_from_blocks(block_list, block_size):
                        grp_idx = gi
                        break
                if grp_idx is None: continue

                slots = self._ordered_slots_for_group(
                    req_state.block_ids[grp_idx],
                    req_state.num_tokens, block_size)
                if not slots: continue

                # Collect Q from buffer in deterministic token order
                q_list, q_sids, tok_idx = [], [], []
                for si, sid in enumerate(slots):
                    q_buf = self.q_buffer.get((layer_idx, sid))
                    if not q_buf: continue
                    q_list.append(q_buf[0]); q_sids.append(sid)
                    tok_idx.append(si)
                if not q_list: continue

                # Read K directly from KV cache
                kv_idx = layer_idx if layer_idx < len(kv_caches) else grp_idx
                if not kv_caches or kv_idx is None or kv_idx >= len(kv_caches): continue
                k_raw = extract_k_from_kv_cache(kv_caches[kv_idx], q_sids)
                k_list = [k_raw[i] for i in range(k_raw.shape[0])]
                if not k_list: continue

                # Filter to compatible Q/K pairs (same shape, device)
                q0, k0 = q_list[0], k_list[0]
                triples = [
                    (idx, q, k) for idx, q, k in zip(tok_idx, q_list, k_list)
                    if (isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor)
                        and q.shape == q0.shape and k.shape == k0.shape
                        and q.device == q0.device and k.device == k0.device) ]
                if not triples: continue
                tok_idx, q_list, k_list = (
                    [t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples])

                # Build tensors and compute attention
                q_t, k_t = torch.stack(q_list), torch.stack(k_list)
                if k_t.is_cuda and not q_t.is_cuda: q_t = q_t.to(k_t.device)

                scale = 1.0 / (q_t.shape[2] ** 0.5)
                attn = self._compute_attention(q_t, k_t, scale)
                if attn is None: continue

                # Apply prefix slice if requested
                if prefix:
                    parts = prefix.split(':')
                    q_start = int(parts[0]) if parts[0] else 0
                    q_end = int(parts[1]) if len(parts) > 1 and parts[1] else None
                    attn = attn[q_start:q_end, :, :]
                    tok_idx = tok_idx[q_start:q_end]

                tmeta = self.build_token_meta(
                    req_state, tok_idx,
                    ordered_slots_len=len(slots))

                # Encode to wire format
                attn = attn.cpu()
                compressed = gzip.compress(attn.numpy().tobytes())
                snapshots.append({
                    'data': base64.b64encode(compressed).decode('utf-8'),
                    'shape': list(attn.shape),
                    'dtype': str(attn.dtype),
                    'layer_idx': layer_idx,
                    'token_meta': tmeta, })

                # Clean up this request's Q slots from buffer
                for sid in set(slots):
                    self.q_buffer.pop((layer_idx, sid), None)

            if snapshots: 
                _shm_write(req_id, snapshots)

        except Exception:
            logger.warning(
                "Capturing attention failed for %s", req_id, exc_info=True)

    def cleanup_request_buffers(
        self, block_ids: list[list[int]], block_size: int,
    ) -> None:
        """Remove buffered Q vectors for a finished request.

        Must be called for ALL finished requests (regardless of capture flag)
        to prevent stale Q data from leaking into future requests that reuse
        the same KV cache blocks.
        """
        if not self.q_buffer or not block_ids: return
        slots_to_remove: Set[int] = set()
        for block_list in block_ids:
            slots_to_remove |= self._slots_from_blocks(block_list, block_size)
        keys_to_remove = [k for k in self.q_buffer if k[1] in slots_to_remove]
        for k in keys_to_remove: 
            del self.q_buffer[k]

    def _extract_layer_idx(self, layer_name: str) -> int:
        
        # cache name
        v = self._layer_idx_cache.get(layer_name) 
        if v is not None: return v
        
        # search if initialzed
        for pat in _LAYER_PATTERNS:
            m = pat.search(layer_name)
            if m: 
                idx = int(m.group(1))
                self._layer_idx_cache[layer_name] = idx
                return idx
            
        # explicitly make error case for error handling
        self._layer_idx_cache[layer_name] = -1
        return -1
