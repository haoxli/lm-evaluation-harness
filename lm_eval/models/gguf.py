import logging
import time

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


logger = logging.getLogger(__name__)


def get_result(logprobs, context_length):
    is_greedy = True
    offsets = logprobs["text_offset"]
    tokens = logprobs["tokens"]
    tokens_logprobs = logprobs["token_logprobs"]

    idx = 0
    while idx < len(offsets) and offsets[idx] < context_length:
        idx += 1
    if idx >= len(tokens_logprobs):
        raise ValueError(
            "Could not locate continuation boundary in logprobs response."
        )
    continuation_logprobs = sum(tokens_logprobs[idx:-1])
    for i in range(idx, len(tokens)):
        token = tokens[i]
        top_tokens = logprobs["top_logprobs"][i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy


@register_model("gguf", "ggml")
class GGUFLM(LM):
    def __init__(self, base_url=None, max_length=2048, **kwargs):
        super().__init__()
        self.base_url = base_url.rstrip("/") if base_url else base_url
        if self.base_url and self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        assert self.base_url, "must pass `base_url` to use GGUF LM!"
        self.logprobs = 10
        self.temperature = 0.0
        self.max_length = max_length
        self._loglikelihood_checked = False
        self._use_fallback = False
        self._n_vocab = None

    def gguf_completion(
        self, context, continuation=None, stop=None, retries=20, delay=5, **kwargs
    ):
        for attempt in range(retries):
            try:
                prompt = context
                request = {
                    "prompt": prompt,
                    "logprobs": self.logprobs,
                    "temperature": self.temperature,
                }
                if continuation:
                    prompt += continuation
                    request.update({"prompt": prompt, "max_tokens": 1, "echo": True})
                if stop is not None:
                    request["stop"] = stop
                response = requests.post(
                    f"{self.base_url}/v1/completions", json=request
                )
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                logger.error(
                    f"RequestException (attempt {attempt + 1}/{retries}): {e}"
                )
                time.sleep(self._backoff(attempt, delay))  # wait before retrying
        else:
            raise RuntimeError(
                f"Failed to get a valid response after {retries} retries."
            )

    @staticmethod
    def _backoff(attempt, delay, cap=60):
        # Exponential backoff capped at `cap` seconds so a transient server
        # error or a server restart does not abort a long-running evaluation.
        return min(cap, delay * (2**attempt))

    def _request_json(self, method, path, payload=None, retries=20, delay=5):
        for attempt in range(retries):
            try:
                response = requests.request(
                    method=method,
                    url=f"{self.base_url}{path}",
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                logger.error(
                    f"RequestException (attempt {attempt + 1}/{retries}) for "
                    f"{path}: {e}"
                )
                time.sleep(self._backoff(attempt, delay))
        raise RuntimeError(f"Failed request to {path} after {retries} retries.")

    def _validate_loglikelihood_support(self):
        if self._loglikelihood_checked:
            return

        probe = self.gguf_completion(context="a", continuation=" b")
        if not (probe and "choices" in probe and probe["choices"]):
            raise ValueError(
                "GGUF loglikelihood preflight failed: server returned no choices."
            )

        probe_logprobs = probe["choices"][0].get("logprobs")
        if probe_logprobs and "content" in probe_logprobs:
            self._use_fallback = True
            logger.info(
                "llama-server b8826+ detected (echo no longer returns prompt logprobs). "
                "Using /completion endpoint with n_probs=200 fast path and full-vocab "
                "fallback for exact loglikelihood scoring."
            )
        self._loglikelihood_checked = True

    def _get_n_vocab(self):
        if self._n_vocab is not None:
            return self._n_vocab

        response = self._request_json("GET", "/v1/models")
        model_data = response.get("data", [])
        if not model_data:
            raise ValueError("Could not read /v1/models response for n_vocab.")

        meta = model_data[0].get("meta", {})
        n_vocab = meta.get("n_vocab")
        if not isinstance(n_vocab, int) or n_vocab <= 0:
            raise ValueError("Missing or invalid n_vocab in /v1/models response.")

        self._n_vocab = n_vocab
        return self._n_vocab

    def _tokenize(self, text):
        response = self._request_json("POST", "/tokenize", {"content": text})
        tokens = response.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("Invalid /tokenize response: missing tokens list.")
        return tokens

    def _detokenize(self, tokens):
        response = self._request_json("POST", "/detokenize", {"tokens": tokens})
        content = response.get("content")
        if not isinstance(content, str):
            raise ValueError("Invalid /detokenize response: missing content string.")
        return content

    def _find_boundary(self, context, full_tokens):
        ctx_tokens = self._tokenize(context)
        if full_tokens[: len(ctx_tokens)] == ctx_tokens:
            return len(ctx_tokens)

        # Fallback for edge cases where tokenization near boundary is ambiguous.
        for idx in range(1, len(full_tokens) + 1):
            prefix_text = self._detokenize(full_tokens[:idx])
            if prefix_text.endswith(context) or prefix_text == context:
                return idx

        raise ValueError(
            "Could not align context/continuation token boundary for loglikelihood."
        )

    def _query_completion(self, prefix_tokens, n_probs):
        response = self._request_json(
            "POST",
            "/completion",
            {
                "prompt": prefix_tokens,
                "n_predict": 1,
                "temperature": 0.0,
                "n_probs": n_probs,
                "cache_prompt": True,
            },
        )
        probs = response.get("completion_probabilities", [])
        if not probs:
            raise ValueError(
                "Invalid /completion response: missing completion_probabilities."
            )
        top_logprobs = probs[0].get("top_logprobs", [])
        if not top_logprobs:
            raise ValueError("Invalid /completion response: missing top_logprobs.")
        return top_logprobs

    def _exact_loglikelihood_via_completion(self, context, continuation):
        full_prompt = context + continuation
        full_tokens = self._tokenize(full_prompt)
        boundary_idx = self._find_boundary(context, full_tokens)
        continuation_tokens = full_tokens[boundary_idx:]

        if not continuation_tokens:
            return 0.0, True

        # N_PROBS_FAST covers virtually all benchmark tokens (pronouns, letters, common
        # words). Only rare tokens trigger the more expensive full-vocab fallback.
        N_PROBS_FAST = 200
        ll = 0.0
        is_greedy = True

        for pos, target_token_id in enumerate(continuation_tokens):
            prefix_tokens = full_tokens[: boundary_idx + pos]
            top_logprobs = self._query_completion(prefix_tokens, N_PROBS_FAST)
            token_to_logprob = {tok["id"]: tok["logprob"] for tok in top_logprobs}

            if target_token_id not in token_to_logprob:
                # Rare token: escalate to full-vocab for exact probability.
                n_vocab = self._get_n_vocab()
                top_logprobs = self._query_completion(prefix_tokens, n_vocab)
                token_to_logprob = {tok["id"]: tok["logprob"] for tok in top_logprobs}
                if target_token_id not in token_to_logprob:
                    raise ValueError(
                        f"Target token id {target_token_id} not found even in "
                        "full-vocab completion probabilities."
                    )

            ll += token_to_logprob[target_token_id]
            if top_logprobs[0].get("id") != target_token_id:
                is_greedy = False

        return ll, is_greedy

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        self._validate_loglikelihood_support()

        res = []
        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            if self._use_fallback:
                try:
                    logprob, is_greedy = self._exact_loglikelihood_via_completion(
                        context=context, continuation=continuation
                    )
                except (RuntimeError, ValueError) as e:
                    # A request can fail persistently (e.g. a prompt that exceeds
                    # the server's context, or an unrecoverable server error after
                    # all retries). Skip it with a safe default so a single bad
                    # request does not abort a multi-hour evaluation.
                    logger.error(
                        "Skipping loglikelihood request after unrecoverable error "
                        f"(returning -inf, is_greedy=False): {e}"
                    )
                    logprob, is_greedy = float("-inf"), False
                res.append((logprob, is_greedy))
                continue

            response = self.gguf_completion(context=context, continuation=continuation)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                if (
                    logprobs
                    and "token_logprobs" in logprobs
                    and logprobs["token_logprobs"]
                ):
                    logprob, is_greedy = get_result(logprobs, len(context))
                    res.append((logprob, is_greedy))
                else:
                    logger.warning(
                        "Invalid logprobs data. Expected 'logprobs' to contain "
                        "'token_logprobs' list. Returning -inf for this request."
                    )
                    res.append((float("-inf"), False))
            else:
                logger.error(
                    f"Invalid response for loglikelihood. Response: {response}. "
                    "Returning -inf for this request."
                )
                res.append((float("-inf"), False))
        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        res = []
        for request in tqdm([req.args for req in requests], disable=disable_tqdm):
            inp = request[0]
            request_args = request[1]
            until = request_args.get("until", ["</s>"])
            response = self.gguf_completion(context=inp, stop=until)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "text" in choice:
                    generated_text = choice["text"].strip()
                    res.append(generated_text)
                else:
                    logger.error(
                        f"Invalid response for greedy_until. Response: {response}"
                    )
                    res.append(None)  # Add default value in case of error
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                res.append(None)  # Add default value in case of error
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )
