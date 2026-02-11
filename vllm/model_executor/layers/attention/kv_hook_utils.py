"""KV Cache Hook Utilities for Post-hoc Attention Analysis

This module provides utilities to capture and analyze attention patterns
after request completion with zero generation overhead.
"""

from dataclasses import dataclass
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

        results = []
        files.sort()
        
        for file_path in files:
            try:
                data = torch.load(file_path)
                attn = data['attn_scores']

                if prefix:
                    parts = prefix.split(':')
                    q_start = int(parts[0]) if parts[0] else 0
                    q_end = int(parts[1]) if len(parts) > 1 and parts[1] else None
                    attn = attn[q_start:q_end, :, :]

                compressed = gzip.compress(attn.numpy().tobytes())
                
                results.append({ 
                    'data': base64.b64encode(compressed).decode('utf-8'),
                    'shape': list(attn.shape), 
                    'dtype': str(attn.dtype),
                    'layer_idx': data.get('layer_idx') 
                })
                
            except Exception as e:
                with open('/tmp/kv_load_debug.txt', 'a') as log:
                    log.write(f"[Load] ERROR loading {file_path}: {e}\n")
                continue
        
        # Delete all files after loading
        for f in files:
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
            query_cpu = query.detach().cpu().clone()
            key_cpu = key.detach().cpu().clone()
        except: 
            return
        
        for i in range(query.shape[0]):
            slot_id = slot_ids[i].item()
            if slot_id < 0:
                continue

            buffer_key = (layer_idx, slot_id)
            if buffer_key not in self.q_buffer: self.q_buffer[buffer_key] = []

            q_token = query_cpu[i].to(torch.float16) if query_cpu[i].dtype != torch.float16 else query_cpu[i]
            self.q_buffer[buffer_key].append(q_token)
            
            if buffer_key not in self.k_buffer: self.k_buffer[buffer_key] = []
            
            k_token = key_cpu[i].to(torch.float16) if key_cpu[i].dtype != torch.float16 else key_cpu[i]
            self.k_buffer[buffer_key].append(k_token)
    
    # May not support Multi-Model, Head-permutation
    def _compute_attention(self, q_tensor, k_tensor, scale):
        q = q_tensor.transpose(0, 1)  # [hq, T, d]
        k = k_tensor.transpose(0, 1)  # [hk, T, d]

        hq, Tq, d = q.shape
        hk, Tk, dk = k.shape
        if d != dk or Tq != Tk:
            return None

        # Always create a mapping index from hk to hq
        if hk == hq:
            k_m = k
        elif hk < hq and (hq % hk == 0):
            # Standard GQA grouping: head group size = hq//hk
            r = hq // hk
            idx = (torch.arange(hq, device=k.device) // r)  # Mapping to 0..hk-1
            k_m = k.index_select(0, idx)
        else:
            # Otherwise (including hk>hq), linear resample: select hq from 0..hk-1
            idx = torch.floor(torch.arange(hq, device=k.device) * (hk / hq)).long()
            idx = torch.clamp(idx, 0, hk - 1)
            k_m = k.index_select(0, idx)

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
            
            # for layer_idx in self.config.layers:
            # Original logic: iterate over target_layers instead of self.config.layers
            for layer_idx in target_layers:  # ← Changed from self.config.layers

                with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                    f.write(f"[Snapshot] Processing layer {layer_idx}\n")
                    f.write(f"[Snapshot] req_state.req_id={req_id}\n")
                    if hasattr(req_state, 'external_req_id'):
                        f.write(f"[Snapshot] req_state.external_req_id={req_state.external_req_id}\n")

                q_list = []
                k_list = []
                for (l_idx, slot_id), q_tokens in self.q_buffer.items():
                    if l_idx == layer_idx: q_list.extend(q_tokens)
                
                for (l_idx, slot_id), k_tokens in self.k_buffer.items():
                    if l_idx == layer_idx: k_list.extend(k_tokens)

                with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                    f.write(f"[Snapshot] q_list: {len(q_list)}, k_list: {len(k_list)}\n")
            
                
                if q_list and k_list:
                    
                    q0 = q_list[0]
                    k0 = k_list[0]
                    q_list = [t for t in q_list if isinstance(t, torch.Tensor) and t.shape == q0.shape and t.device == q0.device]
                    k_list = [t for t in k_list if isinstance(t, torch.Tensor) and t.shape == k0.shape and t.device == k0.device]
                    
                    min_len = min(len(q_list), len(k_list))
                    if min_len == 0: continue
                        
                    q_tensor = torch.stack(q_list[:min_len])  # [T, H, D]
                    k_tensor = torch.stack(k_list[:min_len])  # [T, H, D]
                    
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
                  
                    self.snapshots[req_id] = {
                        'layer_idx': layer_idx,
                        'keys': k_tensor,
                        'queries': q_tensor,
                        'attn_scores': attn_scores,
                    }
                    
                    save_path = f"/tmp/vllm_snapshot_{req_id.replace('-', '_')}_layer{layer_idx}_T{min_len}.pt"
                    torch.save(self.snapshots[req_id], save_path)
                    
                    with open('/tmp/vllm_snapshot_log.txt', 'a') as f:
                        f.write(f"[Snapshot] SUCCESS req={req_id}, layer={layer_idx}, tokens={min_len}\n")
                        f.write(f"[Snapshot] Attn scores: {attn_scores.shape}\n")
                        f.write(f"[Snapshot] SAVED to {save_path}\n")
                        
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