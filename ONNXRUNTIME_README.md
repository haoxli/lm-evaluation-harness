# ONNX Runtime Backend with WebGPU Support

## What is This?

A new backend for lm-evaluation-harness that uses `onnxruntime-genai` with WebGPU execution provider support.

## Quick Commands

### Installation
```bash
pip install onnxruntime-genai onnxruntime-webgpu
```

### Basic Usage
```bash
# Direct lm_eval
lm_eval --model onnxruntime --model_args pretrained="path/to/model" --tasks arc_easy

# Using accuracy runner
python scripts/lm_eval_accuracy_runner.py --backend onnxruntime --tasks arc_easy
```

## Key Features

- ✅ Uses standard `onnxruntime-genai` (not WinML-specific)
- ✅ Supports WebGPU, CUDA, DirectML, CPU execution providers
- ✅ Simple installation (2 packages)
- ✅ Cross-platform compatible
- ✅ Execution provider logging built-in
- ✅ Same accuracy as WinML backend

## Implementation

See the implementation code for details: [lm_eval/models/onnxruntime.py](lm_eval/models/onnxruntime.py)

## Backend Comparison

| Backend | Uses | Best For |
|---------|------|----------|
| `onnxruntime` | `onnxruntime-genai` | WebGPU/CUDA (simple setup) |
| `winml` | `onnxruntime-genai-winml` | Windows NPU (WinML-specific) |

## Verify WebGPU is Active

When you run, check logs for:
```
INFO [models.onnxruntime] ✓ WebGPU execution provider is AVAILABLE
INFO [models.onnxruntime] ✓ Model is CONFIGURED to use WebGPU execution provider
```

## Status

**✅ Production Ready**

- Implementation: Complete
- Testing: Functional
- Documentation: Complete
- WebGPU: Available
- Backend: Registered

## Troubleshooting

**"WebGPU execution provider is not supported"**
```bash
pip install onnxruntime-webgpu
```

**"Module onnxruntime_genai not found"**
```bash
pip install onnxruntime-genai
```

**Check WebGPU availability:**
```bash
python -c "import onnxruntime as ort; print('WebGPU:', 'WebGpuExecutionProvider' in ort.get_available_providers())"
```

## Files

- **Implementation**: [lm_eval/models/onnxruntime.py](lm_eval/models/onnxruntime.py)
- **Accuracy Runner**: [scripts/lm_eval_accuracy_runner.py](../scripts/lm_eval_accuracy_runner.py)
- **Test Script**: [scripts/test_onnxruntime_backend.py](../scripts/test_onnxruntime_backend.py)
