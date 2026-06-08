# LM Evaluation Harness - Agent Instructions

Welcome to the `lm-evaluation-harness` workspace! This file contains conventions and instructions for AI agents working in this codebase. Help AI coding agents understand the workspace and be immediately productive.

## Documentation
- Main repository documentation is located in the [docs/](docs/) folder.
- [docs/README.md](docs/README.md) - Main documentation index.
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) - Contains PR guidelines and code style rules.
- [docs/new_task_guide.md](docs/new_task_guide.md) - Guide for creating and verifying new evaluation tasks.
- [docs/API_guide.md](docs/API_guide.md) and [docs/python-api.md](docs/python-api.md) - API & Library documentation.

## Development and Testing
- **Development Tooling**: The project uses `ruff` for linting. Always install pre-commit hooks via `pre-commit install` after installing dependencies (`pip install -e ".[dev]"`).
- **Run Tests**: 
  ```bash
  python -m pytest --showlocals -s -vv -n=auto --ignore=tests/models/test_openvino.py
  ```
- **Logging**: Enable verbose logging by setting `LMEVAL_LOG_LEVEL="debug"`.

## Project Custom Runner details
- The user has built a custom entrypoint script located at [scripts/lm_eval_runner.py](scripts/lm_eval_runner.py).
- This script simplifies benching across `openvino`, `onnxruntime`, and `llama.cpp` backends using an `auto` detection strategy.
- It parses results from `lm-eval` and automatically aggregates them into flattened `summary.json` and `results.xlsx` spreadsheets. 

## Known Issues and Pitfalls 
- **Windows ONNX WebGPU vs WindowsML**: 
  - `WebGpuExecutionProvider` may be unavailable in some installed ORT GenAI builds on Windows. 
  - Avoid auto-installing `onnxruntime-webgpu` with `onnxruntime-windowsml/onnxruntime-genai-winml` to reduce ORT API mismatch warnings (e.g. `requested API version [24]... only [1, 23]`).
  - When encountering all-NaN logits (non-finite) in `lm-eval` loglikelihood for WebGPU, changing `provider_options` to simply `{ "webgpu": {} }` (removing `validationMode=1` from `genai_config.json`) resolves non-finite failures and restores accuracy.
