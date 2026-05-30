from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests


NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_DEFAULT_MODEL = "qwen/qwen2.5-7b-instruct"
OPENAI_INVOKE_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_DEFAULT_MODEL = "gpt-5_4-mini-2026-03-17"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_BACKEND_FILE = "llm_backend.txt"
logger = logging.getLogger(__name__)
_GLOBAL_RETRYABLE_COOLDOWN_UNTIL = 0.0


class RetryableLLMError(requests.exceptions.RequestException):
    def __init__(self, message: str, *, status_code: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(str(value).strip())
    except Exception:
        return None
    return max(0.0, seconds)


def _extract_retry_after_seconds(response: requests.Response) -> float | None:
    return _parse_retry_after_seconds(response.headers.get("Retry-After"))


def _step_max_tokens(step_name: str, default_max_tokens: int) -> int:
    step = str(step_name or "").strip().lower()
    if step in {"check_whether_roll_dice"}:
        return min(default_max_tokens, 256)
    if step in {"generate_retrieval_queries"}:
        return min(default_max_tokens, 512)
    if step in {
        "pre_response_transition_evaluation",
        "plot_completion_evaluation",
        "scene_completion_evaluation",
        "plot_summary_generation",
        "scene_summary_generation",
    }:
        return min(default_max_tokens, 768)
    if step in {"generate_response", "parser_extract"}:
        return min(default_max_tokens, 1536)
    return default_max_tokens


def _wait_for_global_retryable_cooldown(step_name: str, prompt_length: int) -> None:
    global _GLOBAL_RETRYABLE_COOLDOWN_UNTIL
    remaining = _GLOBAL_RETRYABLE_COOLDOWN_UNTIL - time.monotonic()
    if remaining <= 0:
        return
    logger.info(
        "LLM global cooldown wait step=%s prompt_length=%s sleep_seconds=%.1f",
        step_name,
        prompt_length,
        remaining,
    )
    time.sleep(remaining)


def _extend_global_retryable_cooldown(seconds: float) -> None:
    global _GLOBAL_RETRYABLE_COOLDOWN_UNTIL
    if seconds <= 0:
        return
    _GLOBAL_RETRYABLE_COOLDOWN_UNTIL = max(_GLOBAL_RETRYABLE_COOLDOWN_UNTIL, time.monotonic() + seconds)


def _retry_backoff_seconds(step_name: str, attempt: int, error: Exception) -> float:
    if isinstance(error, RetryableLLMError) and error.retry_after is not None:
        return min(90.0, max(1.0, float(error.retry_after)))

    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return min(60.0, 5.0 * (2 ** (attempt - 1)))
    if status_code in {408, 409, 425}:
        return min(20.0, 2.0 * (2 ** (attempt - 1)))
    return min(20.0, 1.5 * (2 ** (attempt - 1)))


def _parse_key_file(text: str) -> dict[str, str]:
    """
    Supported formats:
    - Single-line bare key (backward compatible): e.g. "nvapi-..."
    - KEY=VALUE lines (recommended):
        NVIDIA_API_KEY=...
        OPENAI_API_KEY=...
    """
    raw = (text or "").strip()
    if not raw:
        return {}

    # Backward-compatible: a single bare key in the file.
    if "\n" not in raw and "=" not in raw:
        return {"NVIDIA_API_KEY": raw}

    out: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            out[k] = v
    return out


def _load_api_key(*, env_var: str, key_filename: str) -> str:
    api_key = (os.getenv(env_var) or "").strip()
    if api_key:
        return api_key

    key_path = PROJECT_ROOT.joinpath(key_filename)
    if not key_path.exists():
        raise ValueError(f"{key_filename} not found in project root and {env_var} is not set")
    text = key_path.read_text(encoding="utf-8")
    parsed = _parse_key_file(text)
    api_key = (parsed.get(env_var) or text.strip()).strip()
    if not api_key or api_key == "PASTE_YOUR_API_KEY_HERE":
        raise ValueError(f"Please set a real API key in {key_filename} or set {env_var}")
    return api_key


def _load_openai_api_key() -> str:
    """
    OpenAI key sources (first match wins — no merge, no conflict):
    1) OPENAI_API_KEY environment variable
    2) openai_api_key.txt (bare sk-... or KEY=VALUE lines)
    3) api_key.txt line OPENAI_API_KEY=... (shared secrets file with NVIDIA)
    """
    env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_key:
        return env_key

    oai_path = PROJECT_ROOT / "openai_api_key.txt"
    if oai_path.exists():
        text = oai_path.read_text(encoding="utf-8")
        parsed = _parse_key_file(text)
        key = (parsed.get("OPENAI_API_KEY") or text.strip()).strip()
        if key and key != "PASTE_YOUR_API_KEY_HERE":
            return key

    shared = PROJECT_ROOT / "api_key.txt"
    if shared.exists():
        text = shared.read_text(encoding="utf-8")
        parsed = _parse_key_file(text)
        key = (parsed.get("OPENAI_API_KEY") or "").strip()
        if key and key != "PASTE_YOUR_API_KEY_HERE":
            return key

    raise ValueError(
        "OpenAI API key not found: set OPENAI_API_KEY, or create openai_api_key.txt, "
        "or add OPENAI_API_KEY=... to api_key.txt"
    )


def _parse_backend_line(raw: str) -> tuple[str, str | None]:
    """
    Parse llm_backend.txt line.
    Supported:
      - "qwen"
      - "openai"
      - "qwen qwen/qwen3.5-397b-a17b"
      - "openai gpt-5_4-mini-2026-03-17"
    """
    text = (raw or "").strip()
    if not text:
        return "", None
    parts = text.split()
    if len(parts) == 1:
        return parts[0].strip().lower(), None
    backend = parts[0].strip().lower()
    model = " ".join(parts[1:]).strip()
    return backend, (model or None)


def _backend_alias_to_provider(token: str) -> str:
    key = (token or "").strip().lower()
    if key in {"qwen", "nvidia", "nv"}:
        return "nvidia"
    if key in {"openai", "oai"}:
        return "openai"
    raise ValueError(
        f"Unsupported LLM backend token {token!r}. Use qwen (NVIDIA), nvidia, or openai "
        f"(see {LLM_BACKEND_FILE} or LLM_PROVIDER)."
    )


def _parse_llm_backend_file() -> tuple[str, dict[str, str]]:
    """
    Parse project-root llm_backend.txt.

    - First non-comment line without ``=``: global default (backend, optional model).
    - Any line containing ``step_name = backend ...``: per-step override (backend + optional model).

    This parser is intentionally tolerant: if there is no explicit global line,
    it falls back to ``qwen`` and still applies per-step overrides.
    """
    path = PROJECT_ROOT / LLM_BACKEND_FILE
    if not path.exists():
        return "qwen", {}
    raw_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        raw_lines.append(s)
    if not raw_lines:
        return "qwen", {}
    global_line = "qwen"
    steps: dict[str, str] = {}
    for line in raw_lines:
        if "=" in line:
            left, right = line.split("=", 1)
            step = left.strip()
            rest = right.strip()
            if step and rest:
                steps[step] = rest
            continue
        # First plain backend line wins as global default.
        if global_line == "qwen":
            global_line = line
    return global_line, steps


def _resolve_provider(
    explicit_provider: str | None,
    step_name: str,
    global_line: str,
    step_map: dict[str, str],
) -> str:
    """Priority: explicit provider > LLM_PROVIDER env > per-step line > global first line > qwen."""
    if explicit_provider is not None and str(explicit_provider).strip():
        backend_token, _ = _parse_backend_line(str(explicit_provider).strip())
        return _backend_alias_to_provider(backend_token)
    env_p = (os.getenv("LLM_PROVIDER") or "").strip()
    if env_p:
        backend_token, _ = _parse_backend_line(env_p)
        return _backend_alias_to_provider(backend_token)
    sk = (step_name or "").strip()
    if sk and sk in step_map:
        backend_token, _ = _parse_backend_line(step_map[sk])
        return _backend_alias_to_provider(backend_token)
    if global_line:
        backend_token, _ = _parse_backend_line(global_line)
        return _backend_alias_to_provider(backend_token)
    return "nvidia"


def _resolve_model(
    provider: str,
    explicit_model: str | None,
    step_name: str,
    global_line: str,
    step_map: dict[str, str],
) -> str:
    """
    Priority: explicit model > NVIDIA_MODEL / OPENAI_MODEL env > per-step model (if step backend matches)
    > global model (if global backend matches) > code default.
    """
    if explicit_model is not None and str(explicit_model).strip():
        return str(explicit_model).strip()
    env_var = "NVIDIA_MODEL" if provider == "nvidia" else "OPENAI_MODEL"
    env_m = (os.getenv(env_var) or "").strip()
    if env_m:
        return env_m
    sk = (step_name or "").strip()
    if sk and sk in step_map:
        b, m = _parse_backend_line(step_map[sk])
        if _backend_alias_to_provider(b) == provider and m:
            return m
    gb, gm = _parse_backend_line(global_line or "")
    if global_line and _backend_alias_to_provider(gb) == provider and gm:
        return gm
    return NVIDIA_DEFAULT_MODEL if provider == "nvidia" else OPENAI_DEFAULT_MODEL


def _resolve_provider_and_model(
    explicit_provider: str | None,
    explicit_model: str | None,
    step_name: str,
) -> tuple[str, str]:
    g_line, step_map = _parse_llm_backend_file()
    provider = _resolve_provider(explicit_provider, step_name, g_line, step_map)
    model = _resolve_model(provider, explicit_model, step_name, g_line, step_map)
    return provider, model


def _openai_chat_uses_max_completion_tokens(model: str) -> bool:
    """
    GPT-5+ (and some reasoning models) reject `max_tokens` on Chat Completions;
    they require `max_completion_tokens` instead.
    """
    if (os.getenv("OPENAI_USE_MAX_COMPLETION_TOKENS") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    m = (model or "").strip().lower()
    return "gpt-5" in m or m.startswith("o1") or m.startswith("o3")


def _call_openai_llm(
    prompt: str,
    model: str,
    *,
    step_name: str,
    max_retries: int,
    timeout: int | float,
) -> str:
    api_key = _load_openai_api_key()

    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", str(max_retries)))
    max_tokens = _step_max_tokens(step_name, int(os.getenv("OPENAI_MAX_TOKENS", "2048")))
    temperature = (
        0.1
        if str(step_name or "").strip().lower() == "branch_transition_decision"
        else float(os.getenv("OPENAI_TEMPERATURE", "0.6"))
    )
    top_p = float(os.getenv("OPENAI_TOP_P", "0.95"))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if _openai_chat_uses_max_completion_tokens(model):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens

    prompt_length = len(prompt)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            _wait_for_global_retryable_cooldown(step_name, prompt_length)
            logger.info(
                "LLM request start provider=openai step=%s attempt=%s/%s prompt_length=%s model=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                model,
            )
            response = requests.post(OPENAI_INVOKE_URL, headers=headers, json=payload, timeout=timeout)
            if not response.ok:
                if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                    raise RetryableLLMError(
                        f"retryable status={response.status_code} body={response.text}",
                        status_code=response.status_code,
                        retry_after=_extract_retry_after_seconds(response),
                    )
                raise ValueError(f"OpenAI request failed ({response.status_code}): {response.text}")
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            logger.warning(
                "LLM request retryable error provider=openai step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except RetryableLLMError as exc:
            last_error = exc
            if exc.status_code == 429:
                _extend_global_retryable_cooldown(_retry_backoff_seconds(step_name, attempt, exc))
            logger.warning(
                "LLM request network error provider=openai step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except requests.exceptions.Timeout as exc:
            last_error = exc
            logger.warning(
                "LLM request timeout provider=openai step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except requests.exceptions.RequestException as exc:
            last_error = exc
            logger.warning(
                "LLM request network error provider=openai step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except Exception as exc:
            logger.error(
                "LLM request failed without retry provider=openai step=%s prompt_length=%s error=%s",
                step_name,
                prompt_length,
                exc,
            )
            raise

        if attempt < max_retries:
            backoff_seconds = _retry_backoff_seconds(step_name, attempt, last_error)
            if getattr(last_error, "status_code", None) == 429:
                _extend_global_retryable_cooldown(backoff_seconds)
            logger.info(
                "LLM request backoff provider=openai step=%s next_attempt=%s sleep_seconds=%.1f",
                step_name,
                attempt + 1,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)

    assert last_error is not None
    logger.error(
        "LLM request exhausted retries provider=openai step=%s prompt_length=%s error=%s",
        step_name,
        prompt_length,
        last_error,
    )
    raise last_error


def _call_nvidia_llm(
    prompt: str,
    model: str,
    *,
    step_name: str,
    max_retries: int,
    timeout: int | float,
) -> str:
    api_key = _load_api_key(env_var="NVIDIA_API_KEY", key_filename="api_key.txt")

    max_retries = int(os.getenv("NVIDIA_MAX_RETRIES", str(max_retries)))
    max_tokens = _step_max_tokens(step_name, int(os.getenv("NVIDIA_MAX_TOKENS", "4096")))
    temperature = (
        0.1
        if str(step_name or "").strip().lower() == "branch_transition_decision"
        else float(os.getenv("NVIDIA_TEMPERATURE", "0.6"))
    )
    top_p = float(os.getenv("NVIDIA_TOP_P", "0.95"))
    top_k = int(os.getenv("NVIDIA_TOP_K", "20"))
    presence_penalty = float(os.getenv("NVIDIA_PRESENCE_PENALTY", "0"))
    repetition_penalty = float(os.getenv("NVIDIA_REPETITION_PENALTY", "1"))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    enable_thinking = os.getenv("NVIDIA_ENABLE_THINKING", "false").lower() in {"1", "true", "yes", "on"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "nvext": {
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
        },
    }

    prompt_length = len(prompt)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            _wait_for_global_retryable_cooldown(step_name, prompt_length)
            logger.info(
                "LLM request start provider=nvidia step=%s attempt=%s/%s prompt_length=%s model=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                model,
            )
            response = requests.post(NVIDIA_INVOKE_URL, headers=headers, json=payload, timeout=timeout)
            if not response.ok:
                if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                    raise RetryableLLMError(
                        f"retryable status={response.status_code} body={response.text}",
                        status_code=response.status_code,
                        retry_after=_extract_retry_after_seconds(response),
                    )
                raise ValueError(f"LLM request failed ({response.status_code}): {response.text}")
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            logger.warning(
                "LLM request retryable error provider=nvidia step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except RetryableLLMError as exc:
            last_error = exc
            if exc.status_code == 429:
                _extend_global_retryable_cooldown(_retry_backoff_seconds(step_name, attempt, exc))
            logger.warning(
                "LLM request network error step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except requests.exceptions.Timeout as exc:
            last_error = exc
            logger.warning(
                "LLM request timeout provider=nvidia step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except requests.exceptions.RequestException as exc:
            last_error = exc
            logger.warning(
                "LLM request network error provider=nvidia step=%s attempt=%s/%s prompt_length=%s error=%s",
                step_name,
                attempt,
                max_retries,
                prompt_length,
                exc,
            )
        except Exception as exc:
            logger.error(
                "LLM request failed without retry provider=nvidia step=%s prompt_length=%s error=%s",
                step_name,
                prompt_length,
                exc,
            )
            raise

        if attempt < max_retries:
            backoff_seconds = _retry_backoff_seconds(step_name, attempt, last_error)
            if getattr(last_error, "status_code", None) == 429:
                _extend_global_retryable_cooldown(backoff_seconds)
            logger.info(
                "LLM request backoff provider=nvidia step=%s next_attempt=%s sleep_seconds=%.1f",
                step_name,
                attempt + 1,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)

    assert last_error is not None
    logger.error(
        "LLM request exhausted retries provider=nvidia step=%s prompt_length=%s error=%s",
        step_name,
        prompt_length,
        last_error,
    )
    raise last_error


def call_llm(
    prompt: str,
    model: str | None = None,
    *,
    provider: str | None = None,
    step_name: str = "generation",
    max_retries: int = 3,
    timeout: int | float = 120,
) -> str:
    provider, model = _resolve_provider_and_model(provider, model, step_name)
    if provider == "nvidia":
        return _call_nvidia_llm(
            prompt,
            model,
            step_name=step_name,
            max_retries=max_retries,
            timeout=timeout,
        )

    return _call_openai_llm(
        prompt,
        model,
        step_name=step_name,
        max_retries=max_retries,
        timeout=timeout,
    )


def call_nvidia_llm(
    prompt: str,
    model: str | None = None,
    *,
    step_name: str = "generation",
    max_retries: int = 3,
    timeout: int | float = 120,
    allow_env_override: bool = True,
) -> str:
    # Backward-compatible entrypoint used across the codebase.
    # Despite the name, it supports multiple backends via llm_backend.txt / LLM_PROVIDER (default: qwen/NVIDIA).
    #
    # Kept for compatibility with older call sites that pass allow_env_override.
    # If False, we avoid overriding the user-supplied model via *_MODEL env vars.
    if not allow_env_override and model is not None:
        resolved, _ = _resolve_provider_and_model(None, None, step_name)
        if resolved == "nvidia":
            os.environ.pop("NVIDIA_MODEL", None)
        else:
            os.environ.pop("OPENAI_MODEL", None)
    return call_llm(
        prompt,
        model=model,
        step_name=step_name,
        max_retries=max_retries,
        timeout=timeout,
    )


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = call_llm(prompt)
    print(response)
    print("--------------------------------")
    response = call_llm(prompt, provider="nvidia", model="qwen/qwen3.5-397b-a17b")
    print(response)

