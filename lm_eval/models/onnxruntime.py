"""
ONNXRuntime GenAI backend for lm-eval-harness with WebGPU/CUDA/CPU support.

This backend leverages onnxruntime-genai to run models with various execution
providers including WebGPU, CUDA, and CPU. It's particularly useful for running
inference with WebGPU acceleration on Windows and Linux.

Example usage:
    lm_eval --model onnxruntime --model_args pretrained=path/to/onnx/model --tasks hellaswag

"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gc

import numpy as np
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model


if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

eval_logger = logging.getLogger(__name__)


@register_model("onnxruntime", "ort")
class ONNXRuntimeGenAI(TemplateLM):
    """
    ONNXRuntime GenAI backend for lm-eval-harness with WebGPU/CUDA/CPU support.

    This model class provides integration with ONNX Runtime GenAI to enable
    evaluation on WebGPU, CUDA, and other hardware-optimized execution providers.
    """

    _DEFAULT_MAX_LENGTH = 2048

    @classmethod
    def create_from_arg_obj(
        cls, arg_dict: dict[str, Any], additional_config: dict[str, Any] | None = None
    ) -> "ONNXRuntimeGenAI":
        """
        Override to properly merge dictionaries and avoid duplicate keyword arguments.

        Args:
            arg_dict: Dictionary containing model arguments
            additional_config: Optional dictionary containing additional configuration

        Returns:
            Instance of ONNXRuntimeGenAI class
        """
        # Merge the dictionaries, with additional_config taking precedence
        merged_config = {**(arg_dict or {})}
        if additional_config:
            # Filter out None values and merge
            filtered_additional = {
                k: v for k, v in additional_config.items() if v is not None
            }
            merged_config.update(filtered_additional)

        return cls(**merged_config)

    def __init__(
        self,
        pretrained: str,
        tokenizer: str | None = None,
        max_length: int | None = None,
        batch_size: int = 1,
        max_batch_size: int = 64,
        **kwargs,
    ) -> None:
        """
        Initialize ONNXRuntime GenAI model.

        Args:
            pretrained: Path to ONNX model file or directory containing model files
            tokenizer: Optional path/name of a HuggingFace tokenizer used for chat
                templating (``--apply_chat_template``). Defaults to ``pretrained``.
            max_length: Maximum sequence length
            batch_size: Batch size for inference
            max_batch_size: Maximum batch size for auto-batching
        """
        super().__init__()

        # Validate and import dependencies
        self._validate_dependencies()

        # Store configuration
        self.pretrained = pretrained
        self._tokenizer_path = tokenizer or pretrained
        self.max_length = max_length or self._DEFAULT_MAX_LENGTH
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.non_finite_policy = str(kwargs.get("non_finite_policy", "error")).lower()
        if self.non_finite_policy not in {"error", "warn"}:
            raise ValueError(
                "Invalid non_finite_policy. Use 'error' (default) or 'warn'."
            )
        self._non_finite_event_count = 0
        self._non_finite_log_limit = int(kwargs.get("non_finite_log_limit", 5))

        # Warn about batch size limitations
        if batch_size != 1 or max_batch_size != 1:
            eval_logger.warning(
                f"ONNXRuntime backend currently only supports batch size 1. "
                f"Requested batch_size={batch_size}, max_batch_size={max_batch_size}. "
                f"Setting both to 1."
            )
            self.batch_size = 1
            self.max_batch_size = 1

        # Load and compile ONNX model
        self._load_and_compile_model(pretrained)

        # HuggingFace tokenizer is loaded lazily, only when chat templating is
        # requested (see apply_chat_template / chat_template / tokenizer_name).
        self._hf_tokenizer = None

        eval_logger.info("ONNXRuntime GenAI model initialized successfully")

    def _handle_non_finite(
        self,
        values: np.ndarray,
        *,
        stage: str,
        tok: int | None = None,
    ) -> None:
        """Guard against non-finite values to avoid silently invalid benchmarks."""
        if np.all(np.isfinite(values)):
            return

        self._non_finite_event_count += 1
        non_finite_count = int(np.size(values) - int(np.isfinite(values).sum()))
        tok_info = f", token={tok}" if tok is not None else ""
        msg = (
            f"Non-finite values detected during {stage}{tok_info}: "
            f"non_finite_count={non_finite_count}, shape={values.shape}. "
            "Benchmark results are unreliable under non-finite logits."
        )

        if self.non_finite_policy == "error":
            raise RuntimeError(msg)

        if self._non_finite_event_count <= self._non_finite_log_limit:
            eval_logger.warning(msg)
        elif self._non_finite_event_count == self._non_finite_log_limit + 1:
            eval_logger.warning(
                "Further non-finite warnings are suppressed; "
                f"total events so far: {self._non_finite_event_count}"
            )

    def _validate_dependencies(self) -> None:
        """
        Validate that required dependencies are available.

        Raises:
            ImportError: If required dependencies are not installed
        """
        try:
            import onnxruntime_genai as og

            self.og = og
            eval_logger.info(f"ONNX Runtime GenAI version: {og.__version__}")
        except ImportError as e:
            raise ImportError(
                "ONNX Runtime GenAI is required for ONNXRuntime backend. "
                "Install with: pip install onnxruntime-genai"
            ) from e

    def _load_and_compile_model(self, model_path: str) -> None:
        """
        Load and optionally compile ONNX model with ONNX Runtime GenAI.

        Args:
            model_path: Path to ONNX model file or directory

        Raises:
            FileNotFoundError: If model path is not found or invalid
            Exception: If model loading fails
        """
        model_path = Path(model_path)

        # Handle different input formats
        if model_path.is_file() and model_path.suffix == ".onnx":
            input_model_path = model_path.parent  # GenAI expects directory
        elif model_path.is_dir():
            input_model_path = model_path
        else:
            raise FileNotFoundError(f"Model path {model_path} not found or invalid")

        # Load model using ONNX Runtime GenAI
        try:
            eval_logger.info(
                f"Loading model with ONNX Runtime GenAI from: {input_model_path}"
            )

            # Check genai_config.json for execution provider configuration
            genai_config_path = input_model_path / "genai_config.json"
            if genai_config_path.exists():
                import json
                config = json.loads(genai_config_path.read_text(encoding="utf-8"))
                provider_options = (
                    config.get("model", {})
                    .get("decoder", {})
                    .get("session_options", {})
                    .get("provider_options", [])
                )
                if provider_options:
                    eval_logger.info(f"Model configured execution providers: {provider_options}")

            # Load model and tokenizer using GenAI
            self.genai_model = self.og.Model(str(input_model_path))
            self.genai_tokenizer = self.og.Tokenizer(self.genai_model)

            eval_logger.info(
                "Model and tokenizer loaded successfully with ONNX Runtime GenAI"
            )

            # Try to get execution provider information
            self._log_execution_provider_info()

            # Store model info
            self.model_path = input_model_path

        except Exception as e:
            eval_logger.error(
                f"Failed to load model with ONNX Runtime GenAI from {input_model_path}: {e}"
            )
            raise

    def _log_execution_provider_info(self) -> None:
        """Log information about the execution provider being used."""
        try:
            # Try to get available execution providers
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            eval_logger.info(f"Available ONNX Runtime execution providers: {available_providers}")

            # Check if WebGPU is available
            if "WebGpuExecutionProvider" in available_providers:
                eval_logger.info("✓ WebGPU execution provider is AVAILABLE")
            else:
                eval_logger.warning("✗ WebGPU execution provider is NOT available")

            # Check if model is likely using WebGPU based on config
            genai_config_path = self.model_path / "genai_config.json"
            if genai_config_path.exists():
                import json
                config = json.loads(genai_config_path.read_text(encoding="utf-8"))
                provider_options = (
                    config.get("model", {})
                    .get("decoder", {})
                    .get("session_options", {})
                    .get("provider_options", [])
                )
                for provider in provider_options:
                    if isinstance(provider, dict):
                        if "webgpu" in provider:
                            eval_logger.info("✓ Model is CONFIGURED to use WebGPU execution provider")
                        elif "dml" in provider:
                            eval_logger.info("Model is configured to use DirectML execution provider")
                        elif "cuda" in provider:
                            eval_logger.info("Model is configured to use CUDA execution provider")
                        elif "cpu" in provider:
                            eval_logger.info("Model is configured to use CPU execution provider")
        except Exception as e:
            eval_logger.debug(f"Could not determine execution provider info: {e}")

    @property
    def eot_token_id(self) -> int:
        """
        Get the end-of-text token ID.

        Returns:
            End-of-text token ID from the GenAI tokenizer
        """
        try:
            # For GenAI tokenizers, try to access EOS from model config or encoding
            if hasattr(self.genai_model, "config"):
                config = self.genai_model.config
                if hasattr(config, "eos_token_id") and config.eos_token_id:
                    if isinstance(config.eos_token_id, list):
                        return config.eos_token_id[0]
                    else:
                        return config.eos_token_id

        except Exception as e:
            eval_logger.warning(
                f"Error getting EOS token ID: {e}, using fallback value 2"
            )

        # Fallback: return a common EOS token ID
        eval_logger.warning("Could not determine EOS token ID, using fallback value 2")
        return 2

    @property
    def prefix_token_id(self) -> int | None:
        """
        Get the prefix token ID (typically BOS token).

        Returns:
            BOS token ID if available, otherwise None
        """
        try:
            # For GenAI tokenizers, try to access BOS from model config or encoding

            # If we have access to the model config
            if hasattr(self.genai_model, "config"):
                config = self.genai_model.config
                if hasattr(config, "bos_token_id") and config.bos_token_id is not None:
                    return config.bos_token_id

            # Check if encoding empty string gives BOS token
            try:
                empty_enc = self.genai_tokenizer.encode("")
                if len(empty_enc) > 0:
                    return empty_enc[0]
            except Exception:
                pass

            return None

        except Exception:
            return None

    @property
    def max_gen_toks(self) -> int:
        """
        Get the maximum number of tokens to generate.

        Returns:
            Maximum generation tokens (default: 1024)
        """
        return min(1024, self.max_length)

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """
        Tokenize string and return token IDs.

        Args:
            string: Input string to tokenize
            left_truncate_len: If provided, truncate from the left to this length
            add_special_tokens: Whether to add special tokens (note: GenAI tokenizer handles this automatically)

        Returns:
            List of token IDs
        """
        # Use GenAI tokenizer for consistency with model inference
        encoding = self.genai_tokenizer.encode(string)

        # Handle left truncation if requested
        if left_truncate_len is not None and len(encoding) > left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs to decode

        Returns:
            Decoded text string
        """
        return self.genai_tokenizer.decode(tokens)

    @property
    def hf_tokenizer(self):
        """Lazily load a HuggingFace tokenizer for chat templating.

        The ONNX Runtime GenAI tokenizer does not expose chat templates, so we
        load a transformers tokenizer from the ``tokenizer`` path (falling back to
        ``pretrained``) to support ``--apply_chat_template``.
        """
        if self._hf_tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "Applying a chat template with the onnxruntime backend requires "
                    "transformers. Install with: pip install transformers"
                ) from e

            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path,
                trust_remote_code=True,
            )
            eval_logger.info(
                f"Loaded HuggingFace tokenizer for chat templating from: "
                f"{self._tokenizer_path}"
            )
        return self._hf_tokenizer

    @property
    def tokenizer_name(self) -> str:
        """Name of the tokenizer, used to fingerprint request caches."""
        return str(self.hf_tokenizer.name_or_path).replace("/", "__")

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        """Return the chat template string for this model's tokenizer."""
        if chat_template is False or chat_template is None:
            return None
        template = getattr(self.hf_tokenizer, "chat_template", None)
        if isinstance(template, dict):
            key = chat_template if isinstance(chat_template, str) else "default"
            return template.get(key) or next(iter(template.values()), None)
        return template

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Apply the tokenizer's chat template to a chat history."""
        if not getattr(self.hf_tokenizer, "chat_template", None):
            raise NotImplementedError(
                f"The tokenizer at '{self._tokenizer_path}' does not define a chat "
                "template, so --apply_chat_template cannot be used with this model."
            )
        return self.hf_tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    def _run_genai_inference_for_full_logits(self, input_text: str) -> np.ndarray:
        """
        Run inference using ONNX Runtime GenAI to get full logits sequence.

        Args:
            input_text: Input text string to compute logits for

        Returns:
            Logits matrix of shape (seq_len, vocab_size) where logits[i] contains
            predictions for the token at position i+1 given tokens[0:i+1]

        Raises:
            Exception: If inference fails
        """
        params = None
        generator = None
        try:
            # Encode input text to tokens
            input_tokens = self.genai_tokenizer.encode(input_text)

            if len(input_tokens) == 0:
                eval_logger.warning("No tokens to process; returning empty array")
                return np.empty((0, 0), dtype=np.float32)

            # Create generator and get full logits in single pass
            params = self.og.GeneratorParams(self.genai_model)
            params.set_search_options(max_length=self.max_length, do_sample=False)
            generator = self.og.Generator(self.genai_model, params)

            # Append all tokens at once
            generator.append_tokens(input_tokens)

            # Get FULL logits using get_output("logits") - this gives all positions!
            full_logits_tensor = generator.get_output("logits")
            logits_array = np.array(full_logits_tensor, dtype=np.float32)

            # Handle different tensor shapes
            if len(logits_array.shape) == 3:  # (batch_size, seq_len, vocab_size)
                logits_matrix = logits_array[0]  # Remove batch dimension
            elif len(logits_array.shape) == 2:  # (seq_len, vocab_size)
                logits_matrix = logits_array
            else:
                raise ValueError(f"Unexpected logits shape: {logits_array.shape}")

            eval_logger.debug(
                f"Full logits shape: {logits_matrix.shape} for {len(input_tokens)} input tokens"
            )
            return logits_matrix

        except Exception as e:
            eval_logger.error(f"GenAI inference failed: {e}")
            raise
        finally:
            # Release native ORT GenAI objects so the KV-cache buffer is
            # reclaimed promptly instead of pooling across calls.
            del generator
            del params

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        """
        Stub implementation - not used since we override loglikelihood directly.
        ONNXRuntime uses the GenAI tokenizer and overrides loglikelihood to work
        with text inputs directly, avoiding tokenization round-trip issues.

        Args:
            requests: List of tokenized requests
            disable_tqdm: Whether to disable progress bar
            override_bs: Optional batch size override

        Returns:
            Empty list (method not used)
        """
        raise NotImplementedError(
            "ONNXRuntime overrides loglikelihood() directly and does not use _loglikelihood_tokens()"
        )

    def loglikelihood(
        self, requests: list["Instance"], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        """
        Compute log-likelihood using ONNX Runtime GenAI with teacher forcing.

        Args:
            requests: List of instances containing (context, continuation) text pairs
            disable_tqdm: Whether to disable progress bar

        Returns:
            List of tuples containing (log_likelihood, is_greedy) for each request
        """
        results = []

        for request in tqdm(
            requests, disable=disable_tqdm, desc="Computing log-likelihoods"
        ):
            context, continuation = request.args

            if len(continuation) == 0:
                results.append((0.0, True))
                continue

            params = None
            gen = None
            try:
                full_text = context + continuation

                # Split the FULL tokenization at the context boundary. context_len
                # and full_tokens MUST come from the same tokenization basis, or the
                # split is off by however the tokenizer treats the context alone
                # (e.g. a leading BOS present in one but not the other).
                full_tokens = self.tok_encode(full_text)
                context_len = len(self.tok_encode(context)) if context else 0
                continuation_tokens = full_tokens[context_len:]

                if len(continuation_tokens) == 0:
                    results.append((0.0, True))
                    continue

                # Create generator for teacher forcing
                params = self.og.GeneratorParams(self.genai_model)
                params.set_search_options(max_length=self.max_length, do_sample=False)
                gen = self.og.Generator(self.genai_model, params)

                # Initialize context: append context tokens to build the initial state
                context_tokens = full_tokens[:context_len]

                if len(context_tokens) > 0:
                    # Append context tokens as 1D numpy array
                    gen.append_tokens(np.array(context_tokens, dtype=np.int32))
                else:
                    # If no context, append BOS if available
                    if self.prefix_token_id is not None:
                        gen.append_tokens(
                            np.array([self.prefix_token_id], dtype=np.int32)
                        )

                # Teacher forcing: iterate through continuation tokens
                total_ll = 0.0
                greedy_ok = True

                for tok in continuation_tokens:
                    # Get logits for next token prediction (only last position)
                    logits = gen.get_logits()
                    logits_array = np.array(logits, dtype=np.float32)

                    # Extract logits vector (get_logits returns last position only)
                    if len(logits_array.shape) == 3:  # (batch, 1, vocab)
                        next_logits = logits_array[0, -1, :]
                    elif len(logits_array.shape) == 2:  # (1, vocab)
                        next_logits = logits_array[-1, :]
                    else:  # (vocab,)
                        next_logits = logits_array

                    self._handle_non_finite(next_logits, stage="next_logits", tok=int(tok))

                    # Compute log probability using numerically stable log_softmax
                    max_logit = np.max(next_logits)
                    exp_logits = np.exp(next_logits - max_logit)
                    log_sum_exp = max_logit + np.log(np.sum(exp_logits))
                    log_prob = next_logits[int(tok)] - log_sum_exp
                    self._handle_non_finite(
                        np.array([log_prob], dtype=np.float32),
                        stage="log_prob",
                        tok=int(tok),
                    )

                    total_ll += float(log_prob)

                    # Check if greedy (argmax matches actual token)
                    if int(np.argmax(next_logits)) != int(tok):
                        greedy_ok = False

                    # Teacher forcing: append the actual token to continue generation
                    gen.append_tokens(np.array([int(tok)], dtype=np.int32))

                results.append((total_ll, greedy_ok))

            except Exception as e:
                if self.non_finite_policy == "error":
                    raise
                eval_logger.warning(f"Failed to compute loglikelihood: {e}")
                import traceback

                eval_logger.debug(traceback.format_exc())
                results.append((0.0, False))
            finally:
                # Explicitly release native ORT GenAI objects so the KV-cache
                # buffer is reclaimed promptly instead of accumulating across
                # requests (the native allocator pools memory aggressively).
                del gen
                del params

        return results

    def loglikelihood_rolling(
        self, requests: list["Instance"], disable_tqdm: bool = False
    ) -> list[float]:
        """
        Compute rolling log-likelihood for perplexity using ONNX Runtime GenAI.
        Uses sliding windows to handle sequences longer than max_length.

        Args:
            requests: List of instances containing text sequences
            disable_tqdm: Whether to disable progress bar

        Returns:
            List of sum of log-likelihood values for each request
        """
        loglikelihoods = []

        for request in tqdm(
            requests, disable=disable_tqdm, desc="Computing rolling log-likelihoods"
        ):
            string = request.args[0]

            # Use sliding window approach for long sequences
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # Prepare windows for loglikelihood (convert to context/continuation text pairs)
            window_requests = []
            for context_tokens, continuation_tokens in rolling_token_windows:
                context_text = self.tok_decode(context_tokens)
                continuation_text = self.tok_decode(continuation_tokens)
                window_requests.append((context_text, continuation_text))

            # Use loglikelihood to compute log-likelihoods for all windows
            # Create Instance objects for the windows
            from lm_eval.api.instance import Instance

            window_instances = [
                Instance(
                    request_type="loglikelihood",
                    doc={},
                    arguments=(ctx, cont),
                    idx=0,
                    metadata={},
                )
                for ctx, cont in window_requests
            ]

            string_nll = self.loglikelihood(
                window_instances,
                disable_tqdm=True,
            )

            # Extract log-likelihoods (discard is_greedy boolean)
            string_nll = [x[0] for x in string_nll]

            # Sum all window log-likelihoods
            total_nll = sum(string_nll)
            loglikelihoods.append(total_nll)

            # Cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), total_nll)

        return loglikelihoods

    def generate_until(
        self, requests: list["Instance"], disable_tqdm: bool = False
    ) -> list[str]:
        """
        Generate text until stopping criteria using ONNX Runtime GenAI.

        Args:
            requests: List of generation requests with context and generation kwargs
            disable_tqdm: Whether to disable progress bar

        Returns:
            List of generated text strings
        """
        if not requests:
            return []

        results = []

        for request in tqdm(requests, disable=disable_tqdm, desc="Generating text"):
            context, gen_kwargs = request.args

            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            until = gen_kwargs.get("until", [])
            temperature = gen_kwargs.get("temperature", 0.0)
            top_p = gen_kwargs.get("top_p", 1.0)
            top_k = gen_kwargs.get("top_k", 50)
            do_sample = gen_kwargs.get("do_sample", False)

            try:
                # Use GenAI generation API
                generated_text = self._run_genai_generation(
                    context,
                    max_gen_toks,
                    until,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                )
                results.append(generated_text)

            except Exception as e:
                eval_logger.warning(f"Generation failed for request: {e}")
                results.append("")

        return results

    def _run_genai_generation(
        self,
        prompt: str,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = False,
    ) -> str:
        """
        Run text generation using ONNX Runtime GenAI.

        Args:
            prompt: Input text to generate from
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences that will stop generation
            temperature: Sampling temperature (0 = greedy, higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (if False, uses greedy decoding)

        Returns:
            Generated text string
        """
        params = None
        generator = None
        try:
            # Encode prompt first so we know total length for max_length
            input_tokens = self.genai_tokenizer.encode(prompt)
            requested_total_max_length = len(input_tokens) + int(max_tokens)
            total_max_length = max(
                len(input_tokens) + 1,
                min(requested_total_max_length, self.max_length),
            )

            # Create generator parameters
            params = self.og.GeneratorParams(self.genai_model)
            # Set generation parameters based on mode
            if do_sample and temperature > 0:
                # Sampling mode: use provided parameters
                params.set_search_options(
                    max_length=total_max_length,
                    top_p=float(top_p),
                    top_k=int(top_k),
                    temperature=float(temperature),
                    repetition_penalty=1.0,
                )
            else:
                # Greedy mode: force temperature to 0 for deterministic output
                params.set_search_options(
                    max_length=total_max_length,
                    top_p=1.0,
                    top_k=1,
                    temperature=0.0,
                    repetition_penalty=1.0,
                )
            # Create generator
            generator = self.og.Generator(self.genai_model, params)
            # Append encoded prompt tokens to the generator
            generator.append_tokens(input_tokens)

            # Generate tokens
            generated_tokens = []
            while not generator.is_done():
                generator.generate_next_token()

                if generator.is_done():
                    break

                # Get the latest generated token
                output_tokens = generator.get_sequence(0)
                base = len(input_tokens) + len(generated_tokens)
                if len(output_tokens) > base:
                    new_token = output_tokens[base]
                    generated_tokens.append(new_token)

                    # Check stopping criteria
                    if len(generated_tokens) >= max_tokens:
                        break

                    # Decode current generation to check stop sequences
                    if stop_sequences:
                        current_text = self.genai_tokenizer.decode(generated_tokens)
                        if any(stop_seq in current_text for stop_seq in stop_sequences):
                            break

            # Decode generated tokens
            if generated_tokens:
                generated_text = self.genai_tokenizer.decode(generated_tokens)

                # Remove stop sequences from the end
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if generated_text.endswith(stop_seq):
                            generated_text = generated_text[: -len(stop_seq)]
                            break

                return generated_text
            else:
                return ""

        except Exception as e:
            eval_logger.error(f"GenAI generation error: {e}")
            return ""
        finally:
            # Release native ORT GenAI objects so the KV-cache buffer is
            # reclaimed promptly instead of pooling across calls.
            del generator
            del params
