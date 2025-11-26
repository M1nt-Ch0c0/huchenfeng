from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import requests
except ImportError as exc:  # pragma: no cover - defensive import
    raise ImportError(
        "The pipeline.llm module requires the 'requests' package. "
        "Install it with 'pip install requests' before using invoke_llm()."
    ) from exc


logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILENAME = "config.llm.json"
DEFAULT_RESPONSE_PATH: Sequence[Any] = ("choices", 0, "message", "content")
DEFAULT_TIMEOUT = 60
DRIVER_DEFAULT_URLS = {
    "openai_chat": "https://api.openai.com/v1/chat/completions",
    "deepseek_chat": "https://api.deepseek.com/v1/chat/completions",
}


class LLMConfigError(RuntimeError):
    """Raised when the llm configuration file is missing or invalid."""


class LLMInvocationError(RuntimeError):
    """Raised after all providers/models/keys fail."""

    def __init__(self, attempts: Sequence[Dict[str, Any]]):
        self.attempts = attempts
        attempt_lines = [
            f"{idx + 1}. provider={a['provider']} model={a['model']} key={a['key_label']} "
            f"error={a['error']}"
            for idx, a in enumerate(attempts)
        ]
        message = "All configured LLM providers failed."
        if attempt_lines:
            message = f"{message}\n" + "\n".join(attempt_lines)
        super().__init__(message)


@dataclass
class ModelConfig:
    name: str
    keys: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    base_url: Optional[str] = None
    response_path: Optional[Sequence[Any]] = None
    jsonable: bool = False


@dataclass
class ProviderConfig:
    name: str
    driver: str
    base_url: str
    timeout: float
    headers: Dict[str, str]
    default_params: Dict[str, Any]
    response_path: Sequence[Any]
    models: List[ModelConfig]
    json_format_field: Optional[str] = None


class LLMClient:
    """
    Loads provider/model/key information from ``config.llm.json`` and attempts
    each combination until one succeeds.
    """

    def __init__(self, config_path: Optional[str | os.PathLike[str]] = None) -> None:
        raw_path = (
            config_path
            if config_path is not None
            else os.environ.get("LLM_CONFIG_PATH", "")
        )
        self.config_path = (
            Path(raw_path)
            if raw_path
            else Path(__file__).with_name(DEFAULT_CONFIG_FILENAME)
        )
        self.providers = self._load_config()

    def reload(self) -> None:
        """Reload configuration from disk."""
        self.providers = self._load_config()

    def invoke(
        self,
        prompt: Optional[str],
        *,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        json_format: Optional[bool] = None,
        **completion_options: Any,
    ) -> str:
        """
        Execute a chat completion request.

        Args:
            prompt: User prompt when ``messages`` is not supplied.
            system_prompt: Optional system instruction inserted before the user prompt.
            messages: Provide fully formed chat messages to override ``prompt``.
            extra_headers: Extra headers merged into each HTTP request.
            json_format: When not ``None``, try to set the provider-specific
                ``json_format`` field if the config declares one.
            completion_options: Arbitrary JSON fields merged into the payload
                (e.g. temperature, max_tokens). ``model`` and ``messages`` are reserved.

        Returns:
            The assistant ``content`` string extracted via ``response_path``.
        """

        if not prompt and not messages:
            raise ValueError("A prompt or messages list is required.")

        for reserved_field in ("model", "messages"):
            if reserved_field in completion_options:
                raise ValueError(
                    f"The '{reserved_field}' field is managed by the configuration. "
                    "Please edit config.llm.json instead of passing it to invoke()."
                )

        resolved_messages = (
            list(messages)
            if messages is not None
            else self._build_messages(prompt, system_prompt)
        )
        extra_headers = extra_headers or {}

        attempts: List[Dict[str, Any]] = []
        for provider in self.providers:
            for model in provider.models:
                for key_idx, api_key in enumerate(model.keys):
                    try:
                        payload = self._build_payload(
                            provider,
                            model,
                            resolved_messages,
                            completion_options,
                            json_format=json_format,
                        )
                        response_json = self._perform_request(
                            provider, model, api_key, payload, extra_headers
                        )
                        content = self._extract_content(
                            provider, model, response_json
                        )
                        logger.debug(
                            "LLM success provider=%s model=%s key=%s",
                            provider.name,
                            model.name,
                            f"#{key_idx + 1}",
                        )
                        return content
                    except Exception as exc:  # noqa: BLE001 - we aggregate the errors
                        error_message = str(exc)
                        logger.warning(
                            "LLM attempt failed provider=%s model=%s key=%s error=%s",
                            provider.name,
                            model.name,
                            f"#{key_idx + 1}",
                            error_message,
                        )
                        attempts.append(
                            {
                                "provider": provider.name,
                                "model": model.name,
                                "key_label": f"#{key_idx + 1}",
                                "error": error_message,
                            }
                        )
                        continue

        raise LLMInvocationError(attempts)

    def _perform_request(
        self,
        provider: ProviderConfig,
        model: ModelConfig,
        api_key: str,
        payload: Dict[str, Any],
        extra_headers: Dict[str, str],
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **provider.headers,
            **extra_headers,
        }
        url = model.base_url or provider.base_url
        try:
            response = requests.post(
                url,
                timeout=provider.timeout,
                headers=headers,
                json=payload,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"HTTP error for {provider.name}/{model.name}: {exc}") from exc

        if response.status_code >= 400:
            snippet = self._safe_response_snippet(response)
            raise RuntimeError(
                f"{provider.name}/{model.name} returned {response.status_code}: {snippet}"
            )

        try:
            return response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - network response
            raise RuntimeError(
                f"{provider.name}/{model.name} response is not JSON: {exc}"
            ) from exc

    @staticmethod
    def _safe_response_snippet(response: requests.Response) -> str:
        text = (response.text or "").strip()
        return text[:280].replace("\n", " ") if text else "no response body"

    @staticmethod
    def _build_messages(
        prompt: Optional[str], system_prompt: Optional[str]
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if prompt:
            messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _build_payload(
        provider: ProviderConfig,
        model: ModelConfig,
        messages: List[Dict[str, str]],
        completion_options: Dict[str, Any],
        *,
        json_format: Optional[bool],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for source in (provider.default_params, model.params, completion_options):
            payload.update(source)
        # These fields are controlled by the pipeline.
        payload["messages"] = messages
        payload["model"] = model.name
        if (
            json_format is not None
            and provider.json_format_field
            and model.jsonable
        ):
            payload[provider.json_format_field] = bool(json_format)
        return payload

    def _extract_content(
        self,
        provider: ProviderConfig,
        model: ModelConfig,
        response_json: Dict[str, Any],
    ) -> str:
        path = (
            model.response_path
            if model.response_path is not None
            else provider.response_path
        )
        try:
            return self._resolve_path(response_json, path)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unable to extract response using path {path} "
                f"for provider={provider.name} model={model.name}"
            ) from exc

    @staticmethod
    def _resolve_path(data: Any, path: Sequence[Any]) -> Any:
        node = data
        for key in path:
            if isinstance(key, int):
                node = node[key]
            else:
                node = node[key]
        return node

    def _load_config(self) -> List[ProviderConfig]:
        if not self.config_path.exists():
            raise LLMConfigError(
                f"LLM config file not found: {self.config_path}. "
                "Create it based on config.sample.json."
            )

        try:
            raw = json.loads(self.config_path.read_text())
        except json.JSONDecodeError as exc:
            raise LLMConfigError(
                f"Failed to parse {self.config_path}: {exc}"
            ) from exc

        providers_data = raw.get("providers")
        if not isinstance(providers_data, list) or not providers_data:
            raise LLMConfigError(
                f"{self.config_path} must contain a non-empty 'providers' list."
            )

        providers: List[ProviderConfig] = []
        for idx, provider_dict in enumerate(providers_data):
            provider = self._parse_provider(provider_dict, idx)
            providers.append(provider)
        return providers

    def _parse_provider(
        self, provider_dict: Dict[str, Any], index: int
    ) -> ProviderConfig:
        name = provider_dict.get("name")
        if not name:
            raise LLMConfigError(
                f"Provider at index {index} missing required field 'name'."
            )
        driver = provider_dict.get("driver", "openai_chat")

        base_url = provider_dict.get("base_url") or DRIVER_DEFAULT_URLS.get(driver)
        if not base_url:
            raise LLMConfigError(
                f"Provider '{name}' is missing 'base_url' and no default is known "
                f"for driver '{driver}'."
            )

        timeout = float(provider_dict.get("timeout", DEFAULT_TIMEOUT))
        headers = self._string_dict(provider_dict.get("headers", {}), f"{name}.headers")
        default_params = dict(provider_dict.get("default_params", {}))
        response_path = tuple(
            provider_dict.get("response_path", DEFAULT_RESPONSE_PATH)
        )
        json_format_field = provider_dict.get("json_format_field")
        if json_format_field is not None and not isinstance(json_format_field, str):
            raise LLMConfigError(
                f"Provider '{name}' json_format_field must be a string when provided."
            )

        models_raw = provider_dict.get("models")
        if not isinstance(models_raw, list) or not models_raw:
            raise LLMConfigError(f"Provider '{name}' must define at least one model.")

        models = [self._parse_model(model, name, idx) for idx, model in enumerate(models_raw)]

        return ProviderConfig(
            name=name,
            driver=driver,
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            default_params=default_params,
            response_path=response_path,
            models=models,
            json_format_field=json_format_field,
        )

    def _parse_model(
        self, model_dict: Dict[str, Any], provider_name: str, index: int
    ) -> ModelConfig:
        name = model_dict.get("name")
        if not name:
            raise LLMConfigError(
                f"Model at index {index} for provider '{provider_name}' "
                "is missing field 'name'."
            )
        keys = self._resolve_keys(
            model_dict.get("keys"),
            provider_name,
            name,
        )
        params = dict(model_dict.get("params", {}))
        base_url = model_dict.get("base_url")
        response_path = tuple(model_dict["response_path"]) if "response_path" in model_dict else None
        jsonable = bool(model_dict.get("jsonable", False))

        return ModelConfig(
            name=name,
            keys=keys,
            params=params,
            base_url=base_url,
            response_path=response_path,
            jsonable=jsonable,
        )

    def _resolve_keys(
        self,
        keys_value: Any,
        provider_name: str,
        model_name: str,
    ) -> List[str]:
        if not isinstance(keys_value, list) or not keys_value:
            raise LLMConfigError(
                f"Model '{model_name}' under provider '{provider_name}' must define "
                "a non-empty 'keys' list."
            )

        resolved: List[str] = []
        for raw in keys_value:
            key = self._normalize_key(raw)
            if key:
                resolved.append(key)

        if not resolved:
            raise LLMConfigError(
                f"No usable API keys found for provider '{provider_name}' model "
                f"'{model_name}'."
            )

        return resolved

    @staticmethod
    def _normalize_key(raw: Any) -> str:
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("$ENV:"):
                env_name = raw.split(":", 1)[1]
                value = os.getenv(env_name, "").strip()
                if not value:
                    raise LLMConfigError(f"Environment variable '{env_name}' is empty.")
                return value
            if not raw:
                raise LLMConfigError("Encountered an empty API key string.")
            return raw

        if isinstance(raw, dict):
            if "env" in raw:
                env_name = raw["env"]
                value = os.getenv(env_name, "").strip()
                if not value:
                    raise LLMConfigError(f"Environment variable '{env_name}' is empty.")
                return value
            if "value" in raw:
                value = str(raw["value"]).strip()
                if not value:
                    raise LLMConfigError("Encountered an empty API key value.")
                return value

        raise LLMConfigError(
            "API keys must be strings, '$ENV:VARIABLE' references, or "
            "{'env': 'VARIABLE'}/{ 'value': 'KEY' } dictionaries."
        )

    @staticmethod
    def _string_dict(value: Any, label: str) -> Dict[str, str]:
        if not value:
            return {}
        if not isinstance(value, dict):
            raise LLMConfigError(f"{label} must be a mapping of strings.")
        result: Dict[str, str] = {}
        for key, val in value.items():
            result[str(key)] = str(val)
        return result


_DEFAULT_CLIENT: Optional[LLMClient] = None


def _get_default_client(reload: bool = False) -> LLMClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None or reload:
        _DEFAULT_CLIENT = LLMClient()
    return _DEFAULT_CLIENT


def invoke_llm(
    prompt: Optional[str],
    *,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    client: Optional[LLMClient] = None,
    config_path: Optional[str | os.PathLike[str]] = None,
    reload_config: bool = False,
    extra_headers: Optional[Dict[str, str]] = None,
    json_format: Optional[bool] = None,
    **completion_options: Any,
) -> str:
    """
    Convenience function that wraps :class:`LLMClient`.

    Either pass a ``client`` instance or let the helper cache one per process.
    """

    if client is not None and config_path is not None:
        raise ValueError("Pass either client or config_path, not both.")

    if client is None:
        if config_path is not None:
            client = LLMClient(config_path=config_path)
        else:
            client = _get_default_client(reload=reload_config)

    return client.invoke(
        prompt,
        system_prompt=system_prompt,
        messages=messages,
        extra_headers=extra_headers,
        json_format=json_format,
        **completion_options,
    )
