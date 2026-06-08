import argparse
import fnmatch
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PROJECT_ROOT = Path(r"d:\workspace\project\llm")
LM_EVALUATION_HARNESS_ROOT = DEFAULT_PROJECT_ROOT / "lm-evaluation-harness"
DEFAULT_MODEL_ROOT = DEFAULT_PROJECT_ROOT / "models"
DEFAULT_OPENVINO_MODELS_ROOT = DEFAULT_MODEL_ROOT / "ov-genai"
DEFAULT_ONNX_MODELS_ROOT = DEFAULT_MODEL_ROOT / "webgpu-prune-lm-head-share-embedding"
DEFAULT_LLAMACPP_MODELS_ROOT = DEFAULT_MODEL_ROOT / "unsloth"
DEFAULT_LLAMACPP_SERVER_BINARY = (
    DEFAULT_PROJECT_ROOT / "benchmarks" / "llama-b8826-bin-win-vulkan-x64" / "llama-server.exe"
)

DEFAULT_TASKS = ["arc_challenge", "winogrande", "mmlu", "hellaswag"]
BACKEND_CHOICES = ["auto", "openvino", "llama.cpp", "onnxruntime"]
DEFAULT_ONNX_PROVIDER = "WebGpuExecutionProvider"
WEBGPU_BACKEND_CHOICES = ["auto", "vulkan", "d3d12"]
NON_FINITE_POLICY_CHOICES = ["error", "warn"]

ONNX_PROVIDER_TO_GENAI_KEY: dict[str, str] = {
    "WebGpuExecutionProvider": "webgpu",
}

# Apply chat template only to model families that typically need chat/instruction formatting.
DEFAULT_CHAT_TEMPLATE_MODEL_PATTERNS = [
    "*instruct*",
    "*deepseek-r1-distill*",
]

@dataclass(frozen=True)
class ModelSpec:
    backend: str
    model_path: Path


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_wildcard_patterns(value: str) -> list[str]:
    return [item.lower() for item in parse_csv(value)]


def append_model_arg(parts: list[str], key: str, value: str | None) -> None:
    if value is not None and value != "":
        parts.append(f"{key}={value}")


def resolve_task_num_fewshot(_task: str, forced_num_fewshot: int | None) -> int | None:
    return forced_num_fewshot


def canonical_backend_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized in {"onnx", "onnxruntime", "ort"}:
        return "onnxruntime"
    if normalized in {"llama", "llama.cpp", "llamacpp"}:
        return "llama.cpp"
    if normalized in {"ov", "openvino"}:
        return "openvino"
    if normalized == "auto":
        return "auto"
    raise ValueError(f"Unsupported backend: {name}")


def normalize_model_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")


def model_requires_chat_template(model_name: str, extra_patterns: list[str]) -> bool:
    model_name_lower = model_name.lower()
    normalized_model_name = normalize_model_name(model_name)
    patterns = [*DEFAULT_CHAT_TEMPLATE_MODEL_PATTERNS, *extra_patterns]
    for pattern in patterns:
        if (
            fnmatch.fnmatch(model_name_lower, pattern)
            or fnmatch.fnmatch(normalized_model_name, pattern)
        ):
            return True
    return False


def detect_backend(path: Path) -> str | None:
    if not path.exists():
        return None

    if path.is_file():
        if path.suffix.lower() == ".gguf":
            return "llama.cpp"
        if path.suffix.lower() == ".onnx":
            return "onnxruntime"
        return None

    if (path / "openvino_model.xml").exists():
        return "openvino"
    if (path / "genai_config.json").exists():
        return "onnxruntime"
    if any(p.is_file() and p.suffix.lower() == ".onnx" for p in path.iterdir()):
        return "onnxruntime"
    if any(p.is_file() and p.suffix.lower() == ".gguf" for p in path.iterdir()):
        return "llama.cpp"
    return None


def discover_models(model_roots: list[Path], backend_filter: str) -> list[ModelSpec]:
    found: list[ModelSpec] = []
    seen: set[str] = set()

    for root in model_roots:
        if not root.exists():
            raise FileNotFoundError(f"Model root does not exist: {root}")

        for candidate in root.rglob("*"):
            if not candidate.is_dir():
                continue
            backend = detect_backend(candidate)
            if backend is None:
                continue
            if backend_filter != "auto" and backend != backend_filter:
                continue
            key = str(candidate.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(ModelSpec(backend=backend, model_path=candidate.resolve()))

    return sorted(found, key=lambda x: (x.backend, str(x.model_path)))


def resolve_gguf_filename(model_path: Path) -> str:
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return model_path.name
    if not model_path.is_dir():
        raise ValueError(f"llama.cpp model path must be a directory or .gguf file: {model_path}")

    candidates = sorted(p.name for p in model_path.glob("*.gguf"))
    if not candidates:
        raise ValueError(f"No .gguf file found in: {model_path}")
    for item in candidates:
        if "q4_k_m" in item.lower():
            return item
    return candidates[0]


def resolve_tokenizer_path(model_path: Path) -> str | None:
    if model_path.is_file():
        parent = model_path.parent
    else:
        parent = model_path

    marker_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    ]
    if any((parent / marker).exists() for marker in marker_files):
        return str(parent)
    return None


def to_model_arg_path(path: Path) -> str:
    # Use forward slashes in model_args to avoid Windows escaping edge cases.
    return path.resolve().as_posix()


def get_venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_checked(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def get_installed_package_version(python_exe: Path, package_name: str) -> str | None:
    proc = subprocess.run(
        [str(python_exe), "-m", "pip", "show", package_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None

    for line in proc.stdout.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return None


def parse_major_minor(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))




def normalize_onnx_provider_name(provider: str) -> str:
    text = provider.strip()
    if not text:
        raise ValueError("ONNX provider name must be non-empty.")

    # Normalize common aliases
    if text.lower() in {"webgpu", "webgpuexecutionprovider"}:
        return "WebGpuExecutionProvider"

    if text == "WebGpuExecutionProvider":
        return text

    raise ValueError(
        f"Unsupported --onnx-provider value: {provider}. Only WebGpuExecutionProvider is supported."
    )


def get_model_genai_provider_keys(model_path: Path) -> set[str]:
    config_path = model_path / "genai_config.json"
    if not config_path.exists():
        return set()

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    provider_options = (
        payload.get("model", {})
        .get("decoder", {})
        .get("session_options", {})
        .get("provider_options", [])
    )

    if not isinstance(provider_options, list):
        return set()

    keys: set[str] = set()
    for item in provider_options:
        if isinstance(item, dict):
            keys.update(str(k).lower() for k in item.keys())
    return keys


def validate_onnx_provider_model_configs(specs: list[ModelSpec], onnx_provider_class: str) -> None:
    expected_key = ONNX_PROVIDER_TO_GENAI_KEY.get(onnx_provider_class, "").lower()
    if not expected_key:
        return

    mismatches: list[str] = []

    for spec in specs:
        if spec.backend != "onnxruntime":
            continue

        provider_keys = get_model_genai_provider_keys(spec.model_path)
        if provider_keys and expected_key not in provider_keys:
            mismatches.append(
                f"{spec.model_path.name}: configured={sorted(provider_keys)}, requested={expected_key}"
            )

    if mismatches:
        raise ValueError(
            "--onnx-provider does not match one or more model genai_config provider_options:\n"
            + "\n".join(mismatches)
        )


def ensure_project_venv(project_root: Path, backend: str) -> Path:
    venv_dir = project_root / ".venv"
    if not venv_dir.exists():
        print(f"Creating virtual environment: {venv_dir}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    venv_python = get_venv_python(venv_dir)
    if not venv_python.exists():
        # When moving workspaces across devices, a stale/incomplete .venv directory may remain.
        # Recreate it so --setup can recover automatically.
        print(f"Detected broken virtual environment (missing Python): {venv_python}")
        print(f"Recreating virtual environment: {venv_dir}")
        shutil.rmtree(venv_dir, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        venv_python = get_venv_python(venv_dir)
        if not venv_python.exists():
            raise FileNotFoundError(f"Could not find venv Python at: {venv_python}")

    print("\nBackend dependency notes:")
    print("- openvino: lm_eval[optimum], openvino, openvino-tokenizers, openvino-genai (GPU)")
    print("- onnxruntime: onnxruntime-genai, onnxruntime-webgpu (WebGPU execution provider)")
    print("    note: onnxruntime-webgpu is installed last (--no-deps) so its WebGPU")
    print("          binaries override the plain onnxruntime pulled in by genai")
    print("- llama.cpp: lm_eval[hf], llama-server (server mode only), gguf")

    run_checked([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    extras: set[str] = set()
    if backend in {"auto", "openvino"}:
        extras.add("optimum")
    if backend in {"auto", "llama.cpp"}:
        extras.add("hf")

    if extras:
        extras_text = ",".join(sorted(extras))
        run_checked(
            [str(venv_python), "-m", "pip", "install", "-e", f".[{extras_text}]"],
            cwd=project_root,
        )
    else:
        run_checked([str(venv_python), "-m", "pip", "install", "-e", "."], cwd=project_root)

    if backend in {"auto", "openvino"}:
        run_checked(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "openvino",
                "openvino-tokenizers",
                "openvino-genai",
            ]
        )

    if backend in {"auto", "onnxruntime"}:
        # For onnxruntime backend: install onnxruntime-genai, then onnxruntime-webgpu.
        #
        # IMPORTANT: install order matters. All three packages (onnxruntime,
        # onnxruntime-genai, onnxruntime-webgpu) share the same `onnxruntime`
        # site-packages directory and overwrite each other's binaries. The plain
        # `onnxruntime` wheel that onnxruntime-genai depends on does NOT include
        # the WebGPU execution provider, so onnxruntime-webgpu must be installed
        # LAST (with --no-deps) to ensure its WebGPU-enabled binaries win.
        run_checked(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "onnxruntime-genai",
            ]
        )
        run_checked(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                "onnxruntime-webgpu",
            ]
        )

    if backend in {"auto", "llama.cpp"}:
        # llama.cpp backend uses llama-server in server mode only (no llama-cpp-python needed)
        run_checked(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "gguf>=0.10.0",
            ]
        )

    run_checked([str(venv_python), "-m", "pip", "install", "openpyxl"])
    run_checked([str(venv_python), "-m", "pip", "install", "hf_xet"])
    return venv_python


def _wait_for_llama_server(host: str, port: int, timeout: float = 120.0) -> None:
    """Poll the llama-server health endpoint until it responds 200 or timeout expires."""
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5)
    raise TimeoutError(
        f"llama-server did not become ready within {timeout}s at {url}. Last error: {last_exc}"
    )


def _is_llama_server_healthy(host: str, port: int, timeout: float = 5.0) -> bool:
    """Return True if the llama-server /health endpoint responds 200."""
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.status == 200
    except Exception:
        return False


def _ensure_llama_server_healthy(
    proc: subprocess.Popen | None,
    host: str,
    port: int,
    start_kwargs: dict,
) -> tuple[subprocess.Popen, str]:
    """Ensure llama-server is alive before a task; restart it if it crashed/unhealthy.

    Returns (process, base_url). Restarts when the process has exited or the
    /health endpoint is not responding, so a server crash during a long
    multi-task run does not invalidate the remaining tasks.
    """
    process_dead = proc is None or proc.poll() is not None
    if not process_dead and _is_llama_server_healthy(host, port):
        return proc, f"http://{host}:{port}"

    if proc is not None:
        reason = "process exited" if process_dead else "health check failed"
        print(f"[llama-server] Unhealthy ({reason}); restarting ...")
        _stop_llama_server(proc)
    else:
        print("[llama-server] Not running; starting ...")

    new_proc, base_url = _start_llama_server(**start_kwargs)
    return new_proc, base_url


def _start_llama_server(
    server_binary: Path,
    gguf_path: Path,
    port: int,
    ngl: int,
    log_file: Path | None = None,
) -> tuple[subprocess.Popen, str]:
    """Start llama-server as a background process and wait until it is ready.

    Returns (process, base_url) where base_url is e.g. 'http://127.0.0.1:8080'.
    Caller is responsible for calling proc.terminate() when done.
    """
    if not server_binary.exists():
        raise FileNotFoundError(
            f"llama-server binary not found: {server_binary}. "
            "Pass --llama-server-binary to specify the correct path."
        )
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {gguf_path}")

    cmd = [
        str(server_binary),
        "-m", str(gguf_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--n-gpu-layers", str(ngl),
        "-fa", "on",
    ]
    print(f"[llama-server] Starting: {' '.join(cmd)}")
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = log_file.open("wb")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        fh.close()  # file stays open via the OS; proc holds the fd
    else:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        _wait_for_llama_server("127.0.0.1", port)
    except TimeoutError:
        proc.terminate()
        raise

    base_url = f"http://127.0.0.1:{port}"
    print(f"[llama-server] Ready at {base_url}")
    return proc, base_url


def _stop_llama_server(proc: subprocess.Popen) -> None:
    """Terminate a llama-server process and wait for it to exit."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("[llama-server] Stopped.")


def resolve_lm_eval_executable(venv_python: Path) -> list[str]:
    scripts_dir = venv_python.parent
    candidates = [
        scripts_dir / "lm_eval.exe",
        scripts_dir / "lm-eval.exe",
    ]
    for exe in candidates:
        if exe.exists():
            return [str(exe)]
    return [str(venv_python), "-m", "lm_eval"]


def build_model_args(
    spec: ModelSpec,
    extra_model_args: str,
    server_url: str,
    non_finite_policy: str,
) -> tuple[str, str]:
    if spec.backend == "openvino":
        parts = [f"pretrained={to_model_arg_path(spec.model_path)}"]
        tokenizer = resolve_tokenizer_path(spec.model_path)
        if tokenizer:
            parts.append(f"tokenizer={tokenizer}")
        if extra_model_args:
            parts.append(extra_model_args.strip(","))
        return "openvino", ",".join(parts)

    if spec.backend == "onnxruntime":
        parts = [f"pretrained={to_model_arg_path(spec.model_path)}"]
        tokenizer = resolve_tokenizer_path(spec.model_path)
        if tokenizer:
            parts.append(f"tokenizer={tokenizer}")
        append_model_arg(
            parts,
            "non_finite_policy",
            non_finite_policy if non_finite_policy != "error" else None,
        )
        if extra_model_args:
            parts.append(extra_model_args.strip(","))
        return "onnxruntime", ",".join(parts)

    # llama.cpp: always use llama-server via the gguf HTTP model type.
    # token_logprobs with echo=True gives exact per-token logprobs for loglikelihood scoring.
    parts = [f"base_url={server_url}"]
    normalized_extra_args = extra_model_args.strip(",")
    if normalized_extra_args:
        parts.append(normalized_extra_args)
    return "gguf", ",".join(parts)


def _to_short_ep(backend: str) -> str:
    mapping = {
        "openvino": "ov",
        "onnxruntime": "ort",
        "llama.cpp": "llama.cpp",
    }
    return mapping.get(backend, backend)


def _split_metric_and_filter(metric_key: str) -> tuple[str, str]:
    if "," in metric_key:
        metric_name, metric_filter = metric_key.split(",", 1)
        return metric_name, metric_filter
    return metric_key, "none"


def parse_lm_eval_file(
    result_json: Path,
    backend: str,
    model_name: str,
    default_batch_size: str,
    default_limit: int,
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    metrics = payload.get("results")
    if not isinstance(metrics, dict):
        return rows

    versions = payload.get("versions")
    if not isinstance(versions, dict):
        versions = {}

    n_shot_map = payload.get("n-shot")
    if not isinstance(n_shot_map, dict):
        n_shot_map = {}

    config = payload.get("config")
    config_num_fewshot: Any = None
    config_batch_size = default_batch_size
    config_limit = default_limit
    if isinstance(config, dict):
        config_num_fewshot = config.get("num_fewshot")
        config_batch_size = str(config.get("batch_size", default_batch_size))
        try:
            config_limit = int(config.get("limit", default_limit))
        except (TypeError, ValueError):
            config_limit = default_limit

    ep_name = _to_short_ep(backend)

    for task, task_metrics in metrics.items():
        if not isinstance(task_metrics, dict):
            continue

        task_version = versions.get(task)
        task_n_shot = n_shot_map.get(task, config_num_fewshot)

        for metric_name, value in task_metrics.items():
            if "_stderr" in metric_name:
                continue
            if not isinstance(value, (int, float)):
                continue

            metric_base, metric_filter = _split_metric_and_filter(metric_name)
            stderr_key = f"{metric_base}_stderr,{metric_filter}"
            stderr_value = task_metrics.get(stderr_key)
            if stderr_value is None:
                stderr_value = task_metrics.get(f"{metric_base}_stderr")

            rows.append(
                {
                    "ep": ep_name,
                    "model_name": model_name,
                    "task": task,
                    "version": task_version,
                    "filter": metric_filter,
                    "n-shot": task_n_shot,
                    "metric": metric_base,
                    "value": value,
                    "stderr": stderr_value,
                    "batch_size": config_batch_size,
                    "limits": config_limit,
                }
            )

    return rows


def collect_summary_rows(
    run_records: list[dict[str, Any]],
    default_batch_size: str,
    default_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in run_records:
        json_candidates: list[Path] = []

        result_json_text = rec.get("result_json", "")
        if result_json_text:
            result_json = Path(result_json_text)
            if result_json.exists():
                json_candidates.append(result_json)

        # Backward-compatible fallback for historical runs without result_json in manifest.
        if not json_candidates:
            result_dir_text = rec.get("result_dir", "")
            if not result_dir_text:
                continue
            result_dir = Path(result_dir_text)
            if not result_dir.exists():
                continue
            for json_file in result_dir.glob("*.json"):
                if json_file.name in {"run_manifest.json", "metrics_flattened.json", "summary.json"}:
                    continue
                json_candidates.append(json_file)

        for json_file in json_candidates:
            rows.extend(
                parse_lm_eval_file(
                    result_json=json_file,
                    backend=str(rec.get("backend", "")),
                    model_name=str(rec.get("model", "")),
                    default_batch_size=default_batch_size,
                    default_limit=default_limit,
                )
            )

    return rows


_PREFERRED_METRICS = ["acc", "acc_norm"]


def pivot_summary_rows_for_excel(
    summary_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Pivot flat per-metric rows into one row per (ep, model, task, filter, n-shot),
    placing each metric value and its stderr as named columns.

    Returns ``(pivoted_rows, ordered_metric_names)`` where *ordered_metric_names*
    lists the unique metric base names encountered, with preferred metrics first.
    """
    grouped: dict[tuple, dict[str, Any]] = {}
    ordered_metrics: list[str] = []
    seen_metrics: set[str] = set()

    for row in summary_rows:
        key = (
            row.get("ep"),
            row.get("model_name"),
            row.get("task"),
            row.get("version"),
            row.get("filter"),
            row.get("n-shot"),
            row.get("batch_size"),
            row.get("limits"),
        )
        if key not in grouped:
            grouped[key] = {
                "ep": row.get("ep"),
                "model_name": row.get("model_name"),
                "task": row.get("task"),
                "version": row.get("version"),
                "filter": row.get("filter"),
                "n-shot": row.get("n-shot"),
                "batch_size": row.get("batch_size"),
                "limits": row.get("limits"),
            }
        metric = row.get("metric", "")
        if metric:
            grouped[key][metric] = row.get("value")
            stderr = row.get("stderr")
            if stderr is not None:
                grouped[key][f"{metric}_stderr"] = stderr
            if metric not in seen_metrics:
                seen_metrics.add(metric)
                ordered_metrics.append(metric)

    preferred = [m for m in _PREFERRED_METRICS if m in seen_metrics]
    others = [m for m in ordered_metrics if m not in _PREFERRED_METRICS]
    return list(grouped.values()), preferred + others


def export_excel(
    run_root: Path,
    run_records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    excel_output: Path | None = None,
) -> Path | None:
    try:
        Workbook = importlib.import_module("openpyxl").Workbook
    except Exception:
        print("openpyxl is not available. Skipping Excel export.")
        return None

    pivoted_rows, ordered_metrics = pivot_summary_rows_for_excel(summary_rows)

    # Build ordered column list: identity fields, then metric+stderr pairs, then tail.
    base_headers = ["ep", "model_name", "task", "version", "filter", "n-shot"]
    metric_headers: list[str] = []
    for metric in ordered_metrics:
        metric_headers.append(metric)
        metric_headers.append(f"{metric}_stderr")
    tail_headers = ["batch_size", "limits"]
    summary_headers = base_headers + metric_headers + tail_headers

    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = "summary"
    ws_summary.append(summary_headers)
    for row in pivoted_rows:
        ws_summary.append([row.get(h) for h in summary_headers])

    ws_runs = wb.create_sheet("runs")
    ws_runs.append(
        [
            "backend",
            "model",
            "task",
            "tasks",
            "num_fewshot",
            "command",
            "exit_code",
            "status",
            "log_file",
            "result_dir",
            "result_json",
        ]
    )
    for rec in run_records:
        task_value = rec.get("task", "")
        tasks_value = rec.get("tasks", [])
        if isinstance(tasks_value, list):
            tasks_text = ",".join(tasks_value)
        else:
            tasks_text = str(tasks_value or "")

        if not task_value and tasks_text:
            task_value = tasks_text.split(",", 1)[0]

        ws_runs.append(
            [
                rec.get("backend", ""),
                rec.get("model", ""),
                task_value,
                tasks_text,
                rec.get("num_fewshot", ""),
                rec.get("command", ""),
                rec.get("exit_code", ""),
                rec.get("status", ""),
                rec.get("log_file", ""),
                rec.get("result_dir", ""),
                rec.get("result_json", ""),
            ]
        )

    excel_path = excel_output if excel_output is not None else run_root / "results.xlsx"
    wb.save(excel_path)
    return excel_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark OpenVINO, llama.cpp, and ONNX Runtime models with lm_eval.exe.\n\n"
            "Recommended commands to run accuracy using standard task arrays:\n"
            "  python scripts/lm_eval_accuracy_runner.py --backend openvino --tasks arc_challenge winogrande hellaswag mmlu\n"
            "  python scripts/lm_eval_accuracy_runner.py --backend llama.cpp --tasks arc_challenge winogrande hellaswag mmlu\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--setup", action="store_true", help="Create .venv and install dependencies.")
    parser.add_argument("--project-root", type=Path, default=LM_EVALUATION_HARNESS_ROOT)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--openvino-models-root", type=Path, default=DEFAULT_OPENVINO_MODELS_ROOT)
    parser.add_argument("--onnxruntime-models-root", type=Path, default=DEFAULT_ONNX_MODELS_ROOT)
    parser.add_argument("--llamacpp-models-root", type=Path, default=DEFAULT_LLAMACPP_MODELS_ROOT)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="auto",
        help=f"Backend to run (choices: {', '.join(BACKEND_CHOICES)}).",
    )
    parser.add_argument("--model", default="", help="Exact model directory name filter.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help=(
            "List of lm-eval task names. "
            "Example: mmlu hellaswag arc_challenge winogrande"
        ),
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=0,
        help="Maximum samples per task; use 0 to skip --limit (full-task evaluation).",
    )
    parser.add_argument("--batch-size", default="1")
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help=(
            "Override num_fewshot for all tasks. If omitted, runner does not pass --num_fewshot "
            "and lm-evaluation-harness task defaults are used."
        ),
    )
    parser.add_argument(
        "--onnx-provider",
        default="",
        help=(
            "ONNX Runtime execution provider (default: WebGpuExecutionProvider). "
            "Only WebGPU is currently supported."
        ),
    )
    parser.add_argument(
        "--webgpu-backend",
        choices=WEBGPU_BACKEND_CHOICES,
        default="auto",
        help=(
            "WebGPU backend to use (auto, vulkan, d3d12). "
            "Sets DAWN_DEBUG_BACKEND environment variable. "
            "Default 'auto' lets Dawn choose the best backend."
        ),
    )
    parser.add_argument(
        "--non-finite-policy",
        choices=NON_FINITE_POLICY_CHOICES,
        default="error",
        help=(
            "How the ONNXRuntime backend should handle non-finite logits during benchmarking. "
            "'error' stops the current run immediately; 'warn' logs and continues."
        ),
    )
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument(
        "--chat-template-mode",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "Chat template strategy for generative tasks:\n"
            "  auto   : Apply only if model matches configured patterns (e.g., *instruct*)\n"
            "           AND task requirements are met (skips multiple-choice loglikelihood tasks).\n"
            "  always : Force apply chat templates to all non-onnxruntime tasks.\n"
            "  never  : Run without chat templates completely."
        ),
    )
    parser.add_argument(
        "--chat-template-model-patterns",
        default="",
        help=(
            "Optional extra model wildcard patterns that should use chat template. "
            "Comma-separated. Example: *qwen*,*chat*"
        ),
    )
    parser.add_argument("--extra-model-args", default="")
    # llama-server options (always used for llama.cpp backend)
    parser.add_argument(
        "--llama-server-binary",
        type=Path,
        default=DEFAULT_LLAMACPP_SERVER_BINARY,
        help=f"Path to llama-server.exe used for llama.cpp backend. Default: {DEFAULT_LLAMACPP_SERVER_BINARY}",
    )
    parser.add_argument(
        "--llama-server-port",
        type=int,
        default=8080,
        help="Port for the managed llama-server (default: 8080).",
    )
    parser.add_argument(
        "--llama-server-ngl",
        type=int,
        default=99,
        help="Number of GPU layers to offload in llama-server (default: 99 = all).",
    )
    parser.add_argument(
        "--llama-server-save-log",
        action="store_true",
        help="Save llama-server output to llama_server.log. Disabled by default to avoid large log files.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=LM_EVALUATION_HARNESS_ROOT / "results")
    parser.add_argument(
        "--postprocess-run-root",
        type=Path,
        default=None,
        help=(
            "Convert an existing run folder to summary.json and results.xlsx, then exit.\n"
            "If run_manifest.json is present it is used; otherwise result JSON files are\n"
            "scanned directly from the folder tree."
        ),
    )
    parser.add_argument(
        "--excel-output",
        type=Path,
        default=None,
        help=(
            "Custom output path for the Excel file (used with --postprocess-run-root or a normal run). "
            "Defaults to <run_root>/results.xlsx."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--export-excel", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.postprocess_run_root is not None:
        run_root = args.postprocess_run_root.resolve()
        manifest_path = run_root / "run_manifest.json"

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_records_raw = manifest.get("records", [])
            if not isinstance(run_records_raw, list):
                raise ValueError(f"Invalid records list in manifest: {manifest_path}")
            run_records = [item for item in run_records_raw if isinstance(item, dict)]
            batch_size = str(manifest.get("batch_size", "1"))
            try:
                limit = int(manifest.get("limit", 0))
            except (TypeError, ValueError):
                limit = 0
        else:
            # No manifest – scan JSON result files directly from the folder tree.
            print(f"run_manifest.json not found in {run_root}; scanning result JSON files.")
            _skip_names = {"run_manifest.json", "metrics_flattened.json", "summary.json"}
            run_records = []
            for json_file in sorted(run_root.rglob("*.json")):
                if json_file.name in _skip_names:
                    continue
                rel_parts = json_file.relative_to(run_root).parts
                backend = rel_parts[0] if len(rel_parts) > 1 else ""
                model = rel_parts[1] if len(rel_parts) > 2 else ""
                run_records.append(
                    {
                        "backend": backend,
                        "model": model,
                        "result_json": str(json_file),
                        "result_dir": str(json_file.parent),
                    }
                )
            batch_size = "1"
            limit = 0

        summary_rows = collect_summary_rows(
            run_records=run_records,
            default_batch_size=batch_size,
            default_limit=limit,
        )
        summary_json_path = run_root / "summary.json"
        summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        print(f"Summary JSON written: {summary_json_path}")

        if args.export_excel:
            excel_path = export_excel(
                run_root, run_records, summary_rows, excel_output=args.excel_output
            )
            if excel_path is not None:
                print(f"Excel summary written: {excel_path}")

        return 0

    project_root = args.project_root.resolve()
    venv_python = get_venv_python(project_root / ".venv")
    backend_filter = canonical_backend_name(args.backend)
    if args.onnx_provider:
        selected_onnx_provider_hint = normalize_onnx_provider_name(args.onnx_provider.strip())
    else:
        selected_onnx_provider_hint = DEFAULT_ONNX_PROVIDER

    if args.setup:
        ensure_project_venv(project_root, backend_filter)
        print("Setup completed.")
        return 0

    if not venv_python.exists():
        if args.dry_run:
            print(
                f"Virtual environment not found at {venv_python}. Using system Python for dry-run launcher resolution."
            )
            venv_python = Path(sys.executable)
        else:
            raise FileNotFoundError(
                f"Missing virtual environment Python at {venv_python}. Run with --setup first."
            )

    if args.model_path is not None and args.model:
        raise ValueError("Use either --model-path or --model, not both.")

    tasks = []
    for t in args.tasks:
        tasks.extend(parse_csv(t))
    if not tasks:
        raise ValueError("At least one task must be provided.")
    chat_template_extra_patterns = parse_wildcard_patterns(args.chat_template_model_patterns)

    lm_eval_cmd_base = resolve_lm_eval_executable(venv_python)

    if args.model_path is not None:
        path = args.model_path.resolve()
        detected = detect_backend(path) if backend_filter == "auto" else backend_filter
        if detected is None:
            raise ValueError(f"Could not detect backend for model path: {path}")
        specs = [ModelSpec(backend=detected, model_path=path)]
    else:
        openvino_models_root = (
            args.openvino_models_root
            if args.openvino_models_root != DEFAULT_OPENVINO_MODELS_ROOT
            else args.models_root / "ov-genai"
        ).resolve()
        onnxruntime_models_root = (
            args.onnxruntime_models_root
            if args.onnxruntime_models_root != DEFAULT_ONNX_MODELS_ROOT
            else args.models_root / "webgpu-prune-lm-head-share-embedding"
        ).resolve()
        llamacpp_models_root = (
            args.llamacpp_models_root
            if args.llamacpp_models_root != DEFAULT_LLAMACPP_MODELS_ROOT
            else args.models_root / "unsloth"
        ).resolve()

        if backend_filter == "openvino":
            roots = [openvino_models_root]
        elif backend_filter == "onnxruntime":
            roots = [onnxruntime_models_root]
        elif backend_filter == "llama.cpp":
            roots = [llamacpp_models_root]
        else:
            roots = [
                openvino_models_root,
                onnxruntime_models_root,
                llamacpp_models_root,
            ]

        specs = discover_models(roots, backend_filter)
        if args.model:
            wanted = args.model.strip().lower()
            specs = [spec for spec in specs if spec.model_path.name.lower() == wanted]

    if not specs:
        print("No models discovered.")
        return 1

    has_onnx_specs = any(spec.backend == "onnxruntime" for spec in specs)

    if has_onnx_specs and args.onnx_provider:
        validate_onnx_provider_model_configs(specs, selected_onnx_provider_hint)

    run_root = (args.output / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Prevent UnicodeEncodeError on Windows when lm_eval prints non-ASCII symbols.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = (
        str(project_root)
        if not env.get("PYTHONPATH")
        else f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
    )

    # Set WebGPU backend if explicitly specified
    if args.webgpu_backend != "auto":
        env["DAWN_DEBUG_BACKEND"] = args.webgpu_backend
        print(f"WebGPU backend forced to: {args.webgpu_backend}")

    run_records: list[dict[str, Any]] = []
    failures = 0

    for spec in specs:
        # Auto-patch tokenizer_config.json to prevent Mistral regex warnings and incorrect tokenization
        # Only apply patch for specific OpenVINO models that require it
        patch_models = ["Qwen3-4B-int4-ov", "gpt-oss-20b-int4-ov"]
        tok_dir = resolve_tokenizer_path(spec.model_path)
        if tok_dir and any(m in spec.model_path.name for m in patch_models):
            try:
                tc_path = Path(tok_dir) / "tokenizer_config.json"
                if tc_path.exists():
                    tc_data = json.loads(tc_path.read_text(encoding="utf-8"))
                    if not tc_data.get("fix_mistral_regex"):
                        tc_data["fix_mistral_regex"] = True
                        tc_path.write_text(json.dumps(tc_data, indent=2), encoding="utf-8")
                        print(f"Auto-patched {tc_path.name} with fix_mistral_regex=True for model {spec.model_path.name}")
            except Exception as e:
                print(f"Warning patching tokenizer_config.json: {e}")

        # Start llama-server for llama.cpp specs.
        server_proc: subprocess.Popen | None = None
        server_url = ""
        server_start_kwargs: dict[str, Any] | None = None
        if spec.backend == "llama.cpp":
            server_url = f"http://127.0.0.1:{args.llama_server_port}"

        if spec.backend == "llama.cpp" and not args.dry_run:
            gguf_file_name = resolve_gguf_filename(spec.model_path)
            gguf_path = spec.model_path / gguf_file_name
            server_log: Path | None = None
            if args.llama_server_save_log:
                spec_result_dir = run_root / spec.backend / spec.model_path.name
                server_log = spec_result_dir / "llama_server.log"
            server_start_kwargs = dict(
                server_binary=args.llama_server_binary,
                gguf_path=gguf_path,
                port=args.llama_server_port,
                ngl=args.llama_server_ngl,
                log_file=server_log,
            )
            server_proc, server_url = _start_llama_server(**server_start_kwargs)

        try:
            model_backend, model_args = build_model_args(
                spec=spec,
                extra_model_args=args.extra_model_args,
                server_url=server_url,
                non_finite_policy=args.non_finite_policy,
            )

            for run_task in tasks:
                run_tasks = [run_task]
                run_num_fewshot = resolve_task_num_fewshot(run_task, args.num_fewshot)
                result_dir = run_root / spec.backend / spec.model_path.name
                result_dir.mkdir(parents=True, exist_ok=True)
                result_json_path = result_dir / f"results_{run_task}.json"

                cmd = [
                    *lm_eval_cmd_base,
                    "run",
                    "--model",
                    model_backend,
                    "--model_args",
                    model_args,
                    "--tasks",
                    *run_tasks,
                    "--batch_size",
                    str(args.batch_size),
                    "--output_path",
                    str(result_json_path),
                ]

                if run_num_fewshot is not None:
                    cmd.extend(["--num_fewshot", str(run_num_fewshot)])
                if args.limit > 0:
                    cmd.extend(["--limit", str(args.limit)])
                if args.log_samples:
                    cmd.append("--log_samples")
                # onnxruntime backend does not implement lm-eval chat template
                # support and can fail if this flag is forced.
                # IMPORTANT: For multiple-choice loglikelihood tasks (arc_challenge, mmlu,
                # winogrande, hellaswag), avoid auto-applying chat templates to preserve
                # comparability with standard accuracy baselines.
                should_apply_chat_template = False
                is_multiple_choice_loglikelihood_task = any(
                    mc in run_task.lower() for mc in ["arc_challenge", "mmlu", "winogrande", "hellaswag"]
                )

                if spec.backend != "onnxruntime" and args.chat_template_mode != "never":
                    if args.chat_template_mode == "always":
                        should_apply_chat_template = True
                    elif args.chat_template_mode == "auto" and not is_multiple_choice_loglikelihood_task:
                        should_apply_chat_template = model_requires_chat_template(
                            spec.model_path.name,
                            chat_template_extra_patterns,
                        )

                if should_apply_chat_template:
                    cmd.append("--apply_chat_template")
                if spec.backend == "openvino":
                    cmd.extend(["--device", "GPU"])
                elif spec.backend == "llama.cpp":
                    # gguf HTTP mode uses llama-server; device/offload is controlled by llama-server flags.
                    pass

                task_slug = run_task
                if run_num_fewshot is not None:
                    log_file = result_dir / f"run_{task_slug}_{run_num_fewshot}shot.log"
                else:
                    log_file = result_dir / f"run_{task_slug}.log"

                print("\n" + "=" * 88)
                print(f"Backend : {spec.backend}")
                print(f"Model   : {spec.model_path}")
                print("Command :")
                print(" ".join(cmd))
                print(f"Log file: {log_file}")

                record: dict[str, Any] = {
                    "backend": spec.backend,
                    "model": spec.model_path.name,
                    "model_path": str(spec.model_path),
                    "task": run_task,
                    "tasks": run_tasks,
                    "num_fewshot": run_num_fewshot,
                    "command": " ".join(cmd),
                    "result_dir": str(result_dir),
                    "result_json": str(result_json_path),
                    "log_file": str(log_file),
                    "apply_chat_template": should_apply_chat_template,
                }

                if args.dry_run:
                    record["status"] = "dry-run"
                    record["exit_code"] = None
                    run_records.append(record)
                    continue

                # Verify llama-server is alive before each task and restart it if it
                # crashed during a previous task. Each task is a separate lm_eval
                # invocation, so a mid-run server crash only affects one task.
                if spec.backend == "llama.cpp" and server_start_kwargs is not None:
                    server_proc, server_url = _ensure_llama_server_healthy(
                        server_proc,
                        "127.0.0.1",
                        args.llama_server_port,
                        server_start_kwargs,
                    )

                with log_file.open("w", encoding="utf-8") as fh:
                    fh.write(f"CMD: {' '.join(cmd)}\n")
                    fh.write(f"Log file: {log_file}\n\n")
                    fh.flush()
                    proc = subprocess.run(
                        cmd,
                        cwd=project_root,
                        env=env,
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )

                executed_cmd = cmd
                executed_log_file = log_file

                record["command"] = " ".join(executed_cmd)
                record["log_file"] = str(executed_log_file)
                record["exit_code"] = proc.returncode
                record["status"] = "ok" if proc.returncode == 0 else "failed"
                run_records.append(record)

                print(f"Status  : {record['status']} (exit code {proc.returncode})")

                if proc.returncode != 0:
                    failures += 1
                    if args.fail_fast:
                        break

                print("Sleeping for 5 seconds ...")
                time.sleep(5)

        finally:
            if server_proc is not None:
                _stop_llama_server(server_proc)

        if args.fail_fast and failures > 0:
            break

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "run_root": str(run_root),
        "backend": backend_filter,
        "tasks": tasks,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "num_fewshot": args.num_fewshot,
        "webgpu_backend": args.webgpu_backend,
        "dry_run": args.dry_run,
        "records": run_records,
    }
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written: {manifest_path}")

    summary_rows = collect_summary_rows(
        run_records=run_records,
        default_batch_size=str(args.batch_size),
        default_limit=int(args.limit),
    )
    summary_json_path = run_root / "summary.json"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"Summary JSON written: {summary_json_path}")

    if args.export_excel and not args.dry_run:
        excel_path = export_excel(run_root, run_records, summary_rows, excel_output=args.excel_output)
        if excel_path is not None:
            print(f"Excel summary written: {excel_path}")

    if failures > 0:
        print(f"Completed with {failures} failure(s).")
        return 1

    print("All runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
