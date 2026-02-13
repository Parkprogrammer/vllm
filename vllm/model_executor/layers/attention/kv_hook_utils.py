"""KV Cache Hook Utilities for Post-hoc Attention Analysis

This module provides utilities to capture and analyze attention patterns
after request completion with zero generation overhead.
"""

from dataclasses import dataclass
from bisect import bisect_left
from typing import Dict, List, Optional, Set, Tuple, Any

import os, re, glob, time
import gzip, base64
import torch

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

def load_kv_snapshot_data(req_id: str, prefix: str | None = None) -> list[dict[str, Any]] | None:
    """
    Load KV snapshot(s) for completed request if KV Hook is enabled.
    """
    try:
        req_id_safe = req_id.replace('-', '_')
        pattern = f"/tmp/vllm_snapshot_{req_id_safe}_*.pt"

        # NOTE(jehyun): Simplified wait logic - just wait for files to be written
        time.sleep(0.5)  # Give time for all files to be written
        files = glob.glob(pattern)
        if not files: return None

        # Check first file to see if capture was requested
        first_data = torch.load(files[0])
        extra_args = first_data.get('extra_args')
        capture_on = bool(extra_args) and str(extra_args.get("kv_hook_capture", "0")) == "1"
        if not capture_on:
            return None

        results, loaded_files = [], [] # Track successfully loaded files
        files.sort()

        for file_path in files:
            try:
                data = torch.load(file_path)
                attn = data['attn_scores']
                token_meta = data.get('token_meta')

                # NOTE(jehyun): Keep load-side unsliced to avoid double slicing.
                # Prefix slicing is handled at snapshot time only.
                _ = prefix

                # compressing for sending attn_score to clinet
                compressed = gzip.compress(attn.numpy().tobytes())

                results.append({
                    'data': base64.b64encode(compressed).decode('utf-8'),
                    'shape': list(attn.shape),
                    'dtype': str(attn.dtype),
                    'layer_idx': data.get('layer_idx'),
                    'token_meta': token_meta,
                })
                loaded_files.append(file_path)  # Mark as successfully loaded

            except Exception as e:
                with open('/tmp/kv_load_debug.txt', 'a') as log:
                    log.write(f"[Load] ERROR loading {file_path}: {e}\n")
                continue

        # Delete only successfully loaded files
        for f in loaded_files:
            try:
                os.remove(f)
                with open('/tmp/kv_load_debug.txt', 'a') as log:
                    log.write(f"[Load] deleted: {f}\n")
            except:
                pass
        
        return results if results else None
        
    except Exception as e:
        with open('/tmp/kv_load_debug.txt', 'a') as f:
            f.write(f"[Load] ERROR for {req_id}: {e}\n")
            import traceback
            f.write(traceback.format_exc())
        return None

@dataclass
class HookConfig:
    """Configuration for KV Cache hook"""
    enabled: bool = False
    prefix: Optional[str] = None # Output path prefix
    layers: Set[int] = None
    heads: Optional[Set[int]] = None  # Head indices (None = all)
    topk: Optional[int] = None  # Top-k values per (token, head)

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
        self.k_buffer: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        # Will follow the shutdown logic here form v1/core/sched/scheduler.py
        if self.config.enabled: print(f"[KV Hook] Initialized \
                                          for layers {sorted(config.layers)}")
        
        # Search the layer name once, keep for the same worker.
        self._layer_idx_cache: Dict[str, int] = {}

        # For tracing snapshot, buffering logic on/off for each batch step
        self.runtime_enabled_this_step = False

    def buffer_qk_pair(self, query: torch.Tensor, key: torch.Tensor, attn_metadata, layer_name: str) -> None:
        """Buffer Query, Key token at attnetion-computation"""
        
        if not self.config.enabled: return
        if attn_metadata is None:  return
        
        layer_idx = self._extract_layer_idx(layer_name)
        # if layer_idx < 0 or layer_idx not in self.config.layers: return
        # NOTE(jehyun): Buffer all layers - filtering happens at snapshot time
        # Original: if layer_idx < 0 or layer_idx not in self.config.layers: return
        if layer_idx < 0: return

        slot_ids = attn_metadata.slot_mapping
        if query.shape[0] != slot_ids.shape[0]: return

        try:
            query_cpu, key_cpu = query.detach().cpu().clone(), key.detach().cpu().clone()
        except: 
            return
        
        for i in range(query.shape[0]):
            
            slot_id = slot_ids[i].item()
            if slot_id < 0: continue

            buffer_key = (layer_idx, slot_id)
            if buffer_key not in self.q_buffer: self.q_buffer[buffer_key] = []

            q_token = query_cpu[i].to(torch.float16) if query_cpu[i].dtype != torch.float16 else query_cpu[i]
            self.q_buffer[buffer_key].append(q_token)
            
            if buffer_key not in self.k_buffer: self.k_buffer[buffer_key] = []
            
            k_token = key_cpu[i].to(torch.float16) if key_cpu[i].dtype != torch.float16 else key_cpu[i]
            self.k_buffer[buffer_key].append(k_token)
    
    # May not support Multi-Model, Head-permutation
    # _compute_ordered_slots_from_request
    def slot_from_request(self, req_state, block_size: int) -> list[int]:
        """Compute ordered slot IDs used by this request from block_ids."""
        
        ordered_slots: list[int] = []
        if not req_state.block_ids: return ordered_slots
        num_tokens = req_state.num_tokens
        
        # block_ids is tuple[list[int], ...] - flatten all block lists
        all_blocks = []
        for block_list in req_state.block_ids: all_blocks.extend(block_list)

        # Compute slot IDs in token order: slot_id = block_id * block_size + offset
        # Dedup slots just in case
        tokens_processed = 0
        seen_slots: set[int] = set()
        for block_id in all_blocks:
            # TODO(jehyun): Need more error-handling here. But moving on for now.
            if tokens_processed >= num_tokens: break

            start_slot = block_id * block_size
            tokens_in_this_block = min(block_size, num_tokens - tokens_processed)

            for offset in range(tokens_in_this_block):
                slot_id = start_slot + offset
                if slot_id in seen_slots: continue
                ordered_slots.append(slot_id)
                seen_slots.add(slot_id)

            tokens_processed += tokens_in_this_block

        return ordered_slots

    def build_token_meta(
        self,
        req_state,
        token_idx: list[int],
        *,
        ordered_slots_len: int | None = None,
        captured_tokens_len: int | None = None,
        window_start_slot: int | None = None,
        block_size: int = 16,
    ) -> dict[str, Any]:
        """Build token mapping metadata for post-hoc client-side alignment.

        Args:
            window_start_slot: The actual starting slot_id of the captured window.
                              Used to compute the absolute token offset.
        """

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
        captured_len = int(
            captured_tokens_len if captured_tokens_len is not None else len(token_idx)
        )

        # NOTE(jehyun): Compute window offset
        # The captured window represents the last N tokens of the sequence
        # where N = ordered_len (number of slots in the captured window)
        window_offset = int(total_len - ordered_len)

        # window_start_slot is provided for debugging/logging only
        _ = window_start_slot

        token_idx_min = int(min(token_idx)) if token_idx else None
        token_idx_max = int(max(token_idx)) if token_idx else None
        prompt_boundary_local = bisect_left(token_idx, prompt_len) if token_idx else None

        # Apply window offset to get absolute token indices
        token_idx_shifted = [int(i) + window_offset for i in token_idx]
        prompt_boundary_with_offset = (
            bisect_left(token_idx_shifted, prompt_len) if token_idx_shifted else None
        )

        return {
            "token_idx": [int(i) for i in token_idx],
            "prompt_len": prompt_len,
            "total_len": total_len,
            "generated_range": {"start": prompt_len, "end": total_len},
            "vision_ranges": vision_ranges,
            "language_ranges": language_ranges,
            # Diagnostic-only fields: keep token_idx semantics unchanged.
            "token_idx_basis": "window_local",
            "ordered_slots_len": ordered_len,
            "captured_tokens_len": captured_len,
            "num_tokens": total_len,
            "num_prompt_tokens": prompt_len,
            "window_offset_candidate": window_offset,  # Now computed from actual slot position
            "window_start_slot": window_start_slot,
            "token_idx_min": token_idx_min,
            "token_idx_max": token_idx_max,
            "prompt_boundary_local": prompt_boundary_local,
            "prompt_boundary_with_offset_candidate": prompt_boundary_with_offset,
        }

    def _compute_attention(self, q_tensor, k_tensor, scale):
        q = q_tensor.transpose(0, 1)  # [hq, T, d]
        k = k_tensor.transpose(0, 1)  # [hk, T, d]

        hq, Tq, d = q.shape
        hk, Tk, dk = k.shape

        # Debug: Log attention computation details
        with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
            f.write(f"[Attention] Q: hq={hq}, Tq={Tq}, d={d} | K: hk={hk}, Tk={Tk}, dk={dk}\n")

        if d != dk or Tq != Tk:
            with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                f.write(f"[Attention] MISMATCH! d={d} vs dk={dk}, Tq={Tq} vs Tk={Tk}\n")
            return None

        # Always create a mapping index from hk to hq
        if hk == hq:
            k_m = k
            mode = "same_heads"
        elif hk < hq and (hq % hk == 0):
            # Standard GQA grouping: head group size = hq//hk
            r = hq // hk
            idx = (torch.arange(hq, device=k.device) // r)  # Mapping to 0..hk-1
            k_m = k.index_select(0, idx)
            mode = f"GQA_r={r}"
        else:
            # Otherwise (including hk>hq), linear resample: select hq from 0..hk-1
            idx = torch.floor(torch.arange(hq, device=k.device) * (hk / hq)).long()
            idx = torch.clamp(idx, 0, hk - 1)
            k_m = k.index_select(0, idx)
            mode = f"resample_hq={hq}_hk={hk}"

        with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
            f.write(f"[Attention] Mode: {mode}\n")

        scores = torch.bmm(q, k_m.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        return probs.transpose(0, 1)  # [hq, T, T] -> [T, hq, T]

    
    def snapshot_keys_immediate(self, req_state, block_size: int, kv_caches, prefix: str | None = None) -> None:
        """
         At the timing of freeing request of vllm-engine,
         create 1 snapshot of attention scores for the requested req_id
        """
        req_id = None
        try:
            req_id = req_state.req_id
            
            # NOTE(jehyun): Determine target layers from request or use default
            # This allows per-request layer selection without modifying global config
            target_layers = self.config.layers  # Default from initialization
            if req_state.sampling_params and req_state.sampling_params.extra_args:
                layers_str = req_state.sampling_params.extra_args.get('kv_hook_layers')
                if layers_str:
                    target_layers = set(int(x.strip()) for x in layers_str.split(','))

            # NOTE(jehyun): Do NOT compute ordered_slots here - it will be computed
            # per-layer after finding the matching cache group for multi-modal requests

            # Debug: Log block_ids info
            with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                f.write(f"[Snapshot] block_size={block_size}\n")
                f.write(f"[Snapshot] num_tokens={req_state.num_tokens}\n")
                f.write(f"[Snapshot] block_ids type: {type(req_state.block_ids)}, len: {len(req_state.block_ids) if req_state.block_ids else 0}\n")
                if req_state.block_ids:
                    for i, block_list in enumerate(req_state.block_ids):
                        f.write(f"[Snapshot] block_ids[{i}]: {block_list[:5]}... (len={len(block_list)})\n")

            # for layer_idx in self.config.layers:
            # Original logic: iterate over target_layers instead of self.config.layers
            for layer_idx in target_layers:  # ← Changed from self.config.layers

                # Debug: Log actual buffer slot_ids for this layer
                buffer_slots_this_layer = [slot_id for (l_idx, slot_id) in self.q_buffer.keys() if l_idx == layer_idx]

                if not buffer_slots_this_layer:
                    with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                        f.write(f"[Snapshot] Layer {layer_idx}: No buffer slots, skipping\n")
                    continue

                # Auto-detect which block_ids group matches the buffer
                matching_group_idx = None
                ordered_slots = []

                if req_state.block_ids:
                    buffer_min, buffer_max = min(buffer_slots_this_layer), max(buffer_slots_this_layer)
                    buffer_start_block = buffer_min // block_size

                    with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                        f.write(f"[Snapshot] Layer {layer_idx}: Buffer slot range [{buffer_min}, {buffer_max}]\n")
                        f.write(f"[Snapshot] Buffer start block: {buffer_start_block}\n")

                        # Check each block_ids group - match by starting block
                        for group_idx, block_list in enumerate(req_state.block_ids):
                            if not block_list: continue

                            # Check if buffer's starting block is in this group
                            start_match = buffer_start_block in block_list
                            group_min_slot = min(block_list) * block_size
                            group_max_slot = (max(block_list) + 1) * block_size - 1

                            f.write(f"[Snapshot]   group[{group_idx}]: blocks {block_list[:3]}...{block_list[-2:]} "
                                   f"→ slots [{group_min_slot}, {group_max_slot}], match={start_match}\n")

                            if start_match and matching_group_idx is None:
                                matching_group_idx = group_idx

                    if matching_group_idx is not None:
                        with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                            f.write(f"[Snapshot] MATCH! Layer {layer_idx} uses cache group {matching_group_idx}\n")

                        # Compute ordered_slots using ONLY the matched group
                        matched_blocks = req_state.block_ids[matching_group_idx]
                        tokens_processed = 0
                        seen_slots = set()

                        for block_id in matched_blocks:
                            if tokens_processed >= req_state.num_tokens: break

                            start_slot = block_id * block_size
                            tokens_in_this_block = min(block_size, req_state.num_tokens - tokens_processed)
                            for offset in range(tokens_in_this_block):
                                slot_id = start_slot + offset
                                if slot_id in seen_slots: continue
                                ordered_slots.append(slot_id)
                                seen_slots.add(slot_id)
                            tokens_processed += tokens_in_this_block

                        with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                            f.write(f"[Snapshot] Computed {len(ordered_slots)} slots from matched group\n")
                            f.write(f"[Snapshot] Sample slots: {ordered_slots[:10]}\n")
                    else:
                        with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                            f.write(f"[Snapshot] WARNING: No matching cache group found for layer {layer_idx}\n")
                        continue
                else:
                    with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                        f.write(f"[Snapshot] WARNING: No block_ids for request {req_id}\n")
                    continue

                request_slot_set = set(ordered_slots)
                q_list = []
                k_list = []

                token_idx: list[int] = []

                # NOTE(jehyun): Build a mapping from slot_id to absolute token position
                # For multi-modal requests, ordered_slots may not start from 0
                # We need to find the actual token index based on slot position
                if not ordered_slots:
                    continue

                min_slot = min(ordered_slots)

                # Collect Q/K in deterministic token order.
                for slot_idx, slot_id in enumerate(ordered_slots):
                    q_tokens, k_tokens = self.q_buffer.get((layer_idx, slot_id)), self.k_buffer.get((layer_idx, slot_id))

                    if not q_tokens or not k_tokens: continue

                    q_list.append(q_tokens[0])
                    k_list.append(k_tokens[0])
                    # Use slot_idx as token position within this window
                    token_idx.append(slot_idx)

                with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                    f.write(f"[Snapshot] Buffer has {len(buffer_slots_this_layer)} slots for layer {layer_idx}\n")
                    f.write(f"[Snapshot] Buffer sample slots: {sorted(buffer_slots_this_layer)[:10]}\n")
                    f.write(f"[Snapshot] q_list: {len(q_list)}, k_list: {len(k_list)}\n")
            
                
                if q_list and k_list:

                    q0, k0 = q_list[0], k_list[0]
                    triples = []
                    
                    # Debug: Log tensor shapes before filtering
                    unique_q_shapes = set(tuple(t.shape) for t in q_list if isinstance(t, torch.Tensor))
                    unique_k_shapes = set(tuple(t.shape) for t in k_list if isinstance(t, torch.Tensor))

                    with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                        f.write(f"[Snapshot] Q unique shapes: {unique_q_shapes}\n")
                        f.write(f"[Snapshot] K unique shapes: {unique_k_shapes}\n")
                        f.write(f"[Snapshot] Q[0] shape: {q0.shape}, K[0] shape: {k0.shape}\n")

                    
                    for idx, q_tok, k_tok in zip(token_idx, q_list, k_list):
                        if (isinstance(q_tok, torch.Tensor) and isinstance(k_tok, torch.Tensor)
                                and q_tok.shape == q0.shape and k_tok.shape == k0.shape
                                and q_tok.device == q0.device and k_tok.device == k0.device):
                            triples.append((idx, q_tok, k_tok))
                    if not triples: continue
                    
                    token_idx, q_list, k_list = [t[0] for t in triples], [t[1] for t in triples], \
                                                        [t[2] for t in triples]

                    # Debug: Log if any tensors were filtered out
                    with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                        f.write(f"[Snapshot] After filtering: q_list={len(q_list)}, k_list={len(k_list)}\n")
                    
                    min_len = min(len(q_list), len(k_list))
                    if min_len == 0: continue
                        
                    # [T, H, D]
                    q_tensor, k_tensor = torch.stack(q_list[:min_len]), torch.stack(k_list[:min_len])
                    
                    # calculate attention
                    # support: GQA, Vanilla Attention
                    # need testing: Sliding-Window, Multi-Modal(Encoder) mixed, permutation
                    head_dim = q_tensor.shape[2]
                    scale = 1.0 / (head_dim ** 0.5)
                    attn_scores = self._compute_attention(q_tensor, k_tensor, scale)

                    if attn_scores is None:
                        with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                            f.write(f"[Snapshot] SKIP req={req_id} (incompatible attention)\n")
                        continue
                    
                    # Can this prefix work?
                    if prefix:
                        parts = prefix.split(':')
                        q_start = int(parts[0]) if parts[0] else 0
                        q_end = int(parts[1]) if len(parts) > 1 and parts[1] else None
                        attn_scores = attn_scores[q_start:q_end, :, :]
                        token_idx = token_idx[q_start:q_end]

                    # Pass the starting slot of this window to compute correct offset
                    window_start_slot = min(ordered_slots) if ordered_slots else None

                    token_meta = self.build_token_meta(
                        req_state,
                        token_idx,
                        ordered_slots_len=len(ordered_slots),
                        captured_tokens_len=len(token_idx),
                        window_start_slot=window_start_slot,
                        block_size=block_size,
                    )

                    # Store extra_args for output_processor to access
                    extra_args = None
                    if req_state.sampling_params and req_state.sampling_params.extra_args:
                        extra_args = req_state.sampling_params.extra_args

                    self.snapshots[req_id] = {
                        'layer_idx': layer_idx,
                        'keys': k_tensor,
                        'queries': q_tensor,
                        'attn_scores': attn_scores,
                        'token_meta': token_meta,
                        'extra_args': extra_args,  # Store for output_processor
                    }

                    save_path = f"/tmp/vllm_snapshot_{req_id.replace('-', '_')}_layer{layer_idx}_T{min_len}.pt"
                    tmp_path = save_path + ".tmp"
                    torch.save(self.snapshots[req_id], tmp_path)
                    os.rename(tmp_path, save_path)  # Atomic rename prevents partial reads

                    with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                        f.write(f"[Snapshot] SUCCESS req={req_id}, layer={layer_idx}, tokens={min_len}\n")
                        f.write(f"[Snapshot] Attn scores: {attn_scores.shape}\n")
                        f.write(f"[Snapshot] SAVED to {save_path}\n")
                        f.write(
                            "[SnapshotSummary] "
                            f"REQ={req_id} "
                            f"LAYER={layer_idx} "
                            f"TOKENS={token_meta['captured_tokens_len']}/{token_meta['ordered_slots_len']}/{token_meta['num_tokens']} "
                            f"OFFSET_CAND={token_meta['window_offset_candidate']} "
                            f"LOCAL_BOUNDARY={token_meta['prompt_boundary_local']} "
                            f"OFFSET_BOUNDARY={token_meta['prompt_boundary_with_offset_candidate']}\n"
                        )

                    # Clean up this request's slots from buffer to prevent contamination
                    for slot_id in request_slot_set:
                        key = (layer_idx, slot_id)
                        self.q_buffer.pop(key, None)
                        self.k_buffer.pop(key, None)

                    # break
        except Exception as e:
            with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                f.write(f"[Snapshot] FAILED req={req_id if req_id else 'unknown'}: {e}\n")
                import traceback
                f.write(traceback.format_exc())
            
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
