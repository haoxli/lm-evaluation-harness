# WebGPU Backend Configuration

The `lm_eval_runner.py` script supports explicit WebGPU backend selection for ONNX Runtime with WebGPU.

## ⚠️ Install Order Matters

`onnxruntime`, `onnxruntime-genai`, and `onnxruntime-webgpu` all share the same
`onnxruntime/` directory in site-packages and overwrite each other's native
binaries (`onnxruntime.dll`, `onnxruntime_pybind11_state.pyd`, etc.).

`onnxruntime-genai` depends on the **plain** `onnxruntime` wheel, which does NOT
include the WebGPU execution provider. If `onnxruntime` is installed *after*
`onnxruntime-webgpu`, its non-WebGPU binaries win and you get:

```
RuntimeError: WebGPU execution provider is not supported in this build.
```

**Fix:** install `onnxruntime-webgpu` LAST, with `--no-deps`, so its
WebGPU-enabled binaries override the plain ones:

```bash
pip install onnxruntime-genai
pip install --force-reinstall --no-deps onnxruntime-webgpu
```

Verify WebGPU is available:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# -> ['WebGpuExecutionProvider', 'CPUExecutionProvider']

python -c "import onnxruntime_genai as og; print(og.__version__)"
# -> 0.14.1   (loads without DLL error)
```

`python scripts/lm_eval_runner.py --setup --backend onnxruntime` already performs
the install in the correct order.

> **Console encoding:** on Windows, `lm_eval` prints a `↑` in the results table
> that the default cp1252 codec can't encode. Set `PYTHONUTF8=1` (the runner does
> this automatically) to avoid a `UnicodeEncodeError`.

---

## Usage

### Let Dawn Auto-Select (Default)
```bash
python scripts/lm_eval_runner.py --backend onnxruntime --tasks mmlu
```

### Force Vulkan Backend
```bash
python scripts/lm_eval_runner.py --backend onnxruntime --tasks mmlu --webgpu-backend vulkan
```

### Force D3D12 Backend (Windows)
```bash
python scripts/lm_eval_runner.py --backend onnxruntime --tasks mmlu --webgpu-backend d3d12
```

## How It Works

The `--webgpu-backend` option sets the `DAWN_DEBUG_BACKEND` environment variable, which tells Dawn which native graphics API to use:

- `--webgpu-backend auto` (default): Let Dawn choose automatically
- `--webgpu-backend vulkan`: Force Vulkan backend
- `--webgpu-backend d3d12`: Force Direct3D 12 backend (Windows only)

## When to Use Vulkan

Use `--webgpu-backend vulkan` when:
- You want consistent behavior across Windows and Linux
- You're testing Vulkan-specific optimizations
- D3D12 drivers are causing issues
- You need to compare performance between backends

## Installation

The setup installs both packages in the correct order:

```bash
python scripts/lm_eval_runner.py --setup --backend onnxruntime
```

This installs:
- `onnxruntime-genai` - ONNX Runtime GenAI library
- `onnxruntime-webgpu` - WebGPU execution provider (installed last, `--no-deps`)

## Verify Backend Selection

When running with a non-auto backend, the script will print:
```
WebGPU backend forced to: vulkan
```

## Troubleshooting

If you encounter issues:

1. **Check Vulkan support**:
   ```bash
   vulkaninfo
   ```

2. **Enable Dawn debug output**:
   ```bash
   set DAWN_DEBUG=1
   python scripts/lm_eval_runner.py --webgpu-backend vulkan ...
   ```

3. **Check available providers in Python**:
   ```python
   import onnxruntime as ort
   print(ort.get_available_providers())
   # Should include 'WebGpuExecutionProvider'
   ```

## Performance Comparison

To compare Vulkan vs D3D12 performance:

```bash
# Run with D3D12 (default)
python scripts/lm_eval_runner.py --backend onnxruntime --tasks hellaswag --limit 100

# Run with Vulkan
python scripts/lm_eval_runner.py --backend onnxruntime --tasks hellaswag --limit 100 --webgpu-backend vulkan
```

The backend used is recorded in the run manifest (`run_manifest.json`) under the `webgpu_backend` field.
