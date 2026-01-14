# Attention Capture Implementation Plan v2
## Query Buffering Approach

## Overview
Store queries during generation (minimal overhead), compute attention at finish (0% generation overhead).

---

## Architecture
```
Generation Loop:
  Decode Step → Save query to request.query_buffer → Continue

Request Finish:
  Collect queries → Get KV cache → Batched Q @ K^T → Save/Return
  → Free KV cache
```

**Memory**: ~2MB per request (256 tokens × 32 heads × 128 dim × 2 bytes)

---

## Implementation

### 1. Request Modification
**File**: `vllm/v1/request.py`
**Location**: `__init__` method
```python
# Add after line 73 (self.events = ...)
self.query_buffer: list[torch.Tensor] = []  # Attention capture
```

**Purpose**: Store query tensors for each decode step

---

### 2. Query Storage Hook
**File**: `vllm/attention/layer.py`
**Locations**: Line 344, 359 (both `self.impl.forward` calls)

**Before**:
```python
self.impl.forward(self, query, key, value, self_kv_cache, attn_metadata, ...)
```

**After**:
```python
self.impl.forward(self, query, key, value, self_kv_cache, attn_metadata, ...)
_maybe_save_query(query, attn_metadata, self.layer_name)
```

**Logic**:
- Check `VLLM_CAPTURE_ATTENTION=1`
- Check decode step (not prefill)
- Check layer filter (`VLLM_CAPTURE_LAYERS`)
- Store `query[:1].cpu()` to request buffer

**Challenge**: Get request object from `forward_context`

---

### 3. Finish Hook
**File**: `vllm/v1/core/sched/scheduler.py`
**Location**: Line 1406 (in `_free_request`, before `self._free_blocks`)

**Before**:
```python
def _free_request(self, request: Request):
    ...
    self._free_blocks(request)
```

**After**:
```python
def _free_request(self, request: Request):
    ...
    _maybe_compute_attention(request, self.kv_cache_manager)
    self._free_blocks(request)
```

**Logic**:
- Check if `request.query_buffer` exists and non-empty
- Get KV cache from `self.kv_cache_manager`
- Stack queries: `[T, 1, H, d] → [T, H, d]`
- Extract keys from KV cache: `[num_blocks, 2, ...] → [L, H, d]`
- Batched matmul: `Q @ K^T` → `[T, H, L]`
- Softmax and save

**Challenge**: Access KV cache before it's freed

---

### 4. Utility Functions
**File**: `vllm/v1/attention/backends/utils.py`
**Location**: End of file

**Functions to add**:
```python
def _maybe_save_query(query, attn_metadata, layer_name):
    """
    Save query to request buffer if capture is enabled.
    
    Steps:
    1. Check VLLM_CAPTURE_ATTENTION=1
    2. Check decode step (attn_metadata.num_decode_tokens > 0)
    3. Filter layers (VLLM_CAPTURE_LAYERS env var)
    4. Get request from forward_context (how?)
    5. Store query[:1].cpu() to request.query_buffer
    """
    pass

def _maybe_compute_attention(request, kv_cache_manager):
    """
    Compute attention from buffered queries at request finish.
    
    Steps:
    1. Check if request.query_buffer exists and non-empty
    2. Stack queries: torch.stack(request.query_buffer) → [T, H, d]
    3. Get KV cache from kv_cache_manager (how?)
    4. Extract keys and reshape
    5. Batched Q @ K^T, scale, softmax
    6. Save to file
    """
    pass

def _extract_keys_from_kv_cache(kv_cache, seq_len):
    """
    Extract key cache and reshape for attention computation.
    
    Args:
        kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_size]
        seq_len: Actual sequence length
    
    Returns:
        Keys: [seq_len, num_heads, head_size]
    """
    pass
```

---

## Open Questions

### Q1: How to get Request object in forward pass?
**Options**:
- A: `forward_context.request_id` → lookup in global dict?
- B: Pass via attn_metadata?
- C: Thread-local storage?

**Need to investigate**: `get_forward_context()` structure

### Q2: How to access KV cache in scheduler?
**Options**:
- A: `kv_cache_manager.get_kv_cache(request)`?
- B: From GPU worker directly?

**Need to investigate**: `KVCacheManager` API

### Q3: Layer name extraction
How to get layer index from `self.layer_name`?
- Use existing: `from vllm.model_executor.models.utils import extract_layer_index`

---

## Environment Variables
```bash
# Enable capture
VLLM_CAPTURE_ATTENTION=1

# Filter specific layers (optional, comma-separated)
VLLM_CAPTURE_LAYERS=39  # Last layer only

# Output path (optional)
VLLM_ATTN_OUTPUT=/tmp/attn_weights.pt
```

---

## Testing Plan

### Phase 1: Query Storage
```python
# Test that queries are stored
request = ...
assert len(request.query_buffer) == num_generated_tokens
```

### Phase 2: Attention Computation
```python
# Test batched computation
queries = torch.randn(10, 32, 128)  # 10 tokens, 32 heads, 128 dim
keys = torch.randn(50, 32, 128)     # 50 context tokens
attn = compute_batched_attention(queries, keys)
assert attn.shape == (10, 32, 50)
```

### Phase 3: End-to-End
```bash
VLLM_CAPTURE_ATTENTION=1 \
VLLM_CAPTURE_LAYERS=39 \
python test_capture.py
```

---

## Advantages Over v1

| Aspect | v1 (Post-hoc) | v2 (Buffering) |
|--------|---------------|----------------|
| Generation overhead | 2× compute | ~0% (memory copy) |
| Memory | 0 | 2MB per request |
| Compute timing | Each capture | At finish only |
| Scalability | ✗ (2× per token) | ✓ (batched) |

---

## Next Steps

1. Investigate Q1: Request access in forward pass
2. Investigate Q2: KV cache access in scheduler
3. Implement utility functions
4. Add hooks to 3 files
5. Test with facebook/opt-125m
6. Scale to Qwen3-VL

---

## Files Modified Summary

1. `vllm/v1/request.py` - 1 line added
2. `vllm/attention/layer.py` - 2 lines added (2 locations)
3. `vllm/v1/core/sched/scheduler.py` - 1 line added
4. `vllm/v1/attention/backends/utils.py` - 3 functions added (~80 lines)

**Total**: 4 files, ~85 lines added
