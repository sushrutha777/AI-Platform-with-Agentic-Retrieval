"""Google Cloud Model Armor integration for application boundaries.

Model Armor is deliberately kept outside the retrieval and LLM gateway
layers.  This module only screens the latest user message and the completed
application response.  It does not create or orchestrate any LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.logging import logger


@dataclass
class GuardrailResult:
    """Application-level result returned by a Model Armor scan.

    ``metadata`` contains only safe, structural information such as enum
    values and filter keys.  Raw Model Armor responses are intentionally not
    exposed to callers or users.
    """

    allowed: bool = True
    reason: str = "allowed"
    sanitized_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: bool = False

    @property
    def is_blocked(self) -> bool:
        """Backward-compatible name used by older call sites."""
        return not self.allowed

    @property
    def blocked_reason(self) -> str:
        """Backward-compatible reason accessor."""
        return self.reason if not self.allowed else ""

    @property
    def filter_results(self) -> Dict[str, Any]:
        """Backward-compatible metadata accessor."""
        return self.metadata


# Retain the old public name for consumers that imported it directly.
SanitizeResult = GuardrailResult


class _GuardrailConfigurationError(Exception):
    """Internal marker for invalid Model Armor configuration."""


class ModelArmorGuardrail:
    """Screen user prompts and completed model responses with one template.

    The client is created lazily and reused for the lifetime of the process.
    Authentication is delegated to Google Application Default Credentials,
    which is the supported Cloud Run service-account flow.

    When Model Armor is disabled, calls are explicit passthroughs for local
    development.  When it is enabled, configuration, client, API, timeout,
    and malformed-response failures fail closed so protected requests never
    silently bypass the configured policy.
    """

    def __init__(self, client: Optional[Any] = None) -> None:
        self.enabled = bool(getattr(settings, "MODEL_ARMOR_ENABLED", False))
        self.project_id = getattr(settings, "MODEL_ARMOR_PROJECT_ID", None)
        self.location = getattr(settings, "MODEL_ARMOR_LOCATION", None)
        self.template_id = getattr(settings, "MODEL_ARMOR_TEMPLATE_ID", None)
        self._client = client
        self._client_lock = threading.Lock()
        self._client_initialization_error: Optional[str] = None
        self._configuration_error = False

        timeout_value = getattr(settings, "MODEL_ARMOR_TIMEOUT_SECONDS", 5.0)
        try:
            self.timeout_seconds = float(timeout_value)
        except (TypeError, ValueError):
            self.timeout_seconds = 5.0

        if self.enabled and not all((self.project_id, self.location, self.template_id)):
            self._configuration_error = True
            logger.error(
                "MODEL_ARMOR_ERROR stage=initialization reason=invalid_configuration"
            )

    @property
    def template_name(self) -> str:
        """Return the configured template resource name."""
        return (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/templates/{self.template_id}"
        )

    def _get_client(self) -> Any:
        """Create the regional v1 client once, using Application Default Credentials."""
        if self._configuration_error:
            raise _GuardrailConfigurationError
        if self._client is not None:
            return self._client
        if self._client_initialization_error is not None:
            raise RuntimeError("Model Armor client initialization failed")

        with self._client_lock:
            if self._client is not None:
                return self._client
            if self._client_initialization_error is not None:
                raise RuntimeError("Model Armor client initialization failed")

            try:
                from google.api_core.client_options import ClientOptions
                from google.cloud import modelarmor_v1

                self._client = modelarmor_v1.ModelArmorClient(
                    transport="rest",
                    client_options=ClientOptions(
                        api_endpoint=f"modelarmor.{self.location}.rep.googleapis.com"
                    ),
                )
                return self._client
            except Exception as exc:
                self._client_initialization_error = type(exc).__name__
                raise RuntimeError("Model Armor client initialization failed") from exc

    @staticmethod
    def _enum_name(value: Any) -> str:
        """Return a stable enum name without serializing a raw API response."""
        return getattr(value, "name", str(value))

    def _failure(
        self,
        stage: str,
        reason: str,
        *,
        error_type: Optional[str] = None,
    ) -> GuardrailResult:
        logger.error(
            "MODEL_ARMOR_ERROR stage=%s reason=%s error_type=%s",
            stage,
            reason,
            error_type or "none",
        )
        return GuardrailResult(
            allowed=False,
            reason=reason,
            sanitized_text="",
            error=True,
        )

    def _parse_response(self, response: Any, original_text: str, stage: str) -> GuardrailResult:
        """Parse the v1 ``SanitizationResult`` and enforce fail-closed semantics."""
        try:
            from google.cloud import modelarmor_v1

            sanitization_result = response.sanitization_result
            if sanitization_result is None:
                return self._failure(stage, "malformed_response", error_type="missing_result")

            filter_match_state = sanitization_result.filter_match_state
            invocation_result = sanitization_result.invocation_result
            filter_names = list(getattr(sanitization_result.filter_results, "keys", lambda: [])())
            metadata = {
                "filter_match_state": self._enum_name(filter_match_state),
                "invocation_result": self._enum_name(invocation_result),
                "filter_names": filter_names,
            }

            if filter_match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
                logger.warning(
                    "MODEL_ARMOR_%s_BLOCKED reason=policy_violation",
                    stage.upper(),
                )
                return GuardrailResult(
                    allowed=False,
                    reason="policy_violation",
                    metadata=metadata,
                )

            if filter_match_state != modelarmor_v1.FilterMatchState.NO_MATCH_FOUND:
                return self._failure(stage, "malformed_response", error_type="unknown_match_state")

            if invocation_result != modelarmor_v1.InvocationResult.SUCCESS:
                return self._failure(stage, "scan_incomplete", error_type="incomplete_invocation")

            logger.info("MODEL_ARMOR_%s_ALLOWED", stage.upper())
            # The current API returns a verdict, not a redacted text field.
            return GuardrailResult(
                allowed=True,
                reason="allowed",
                sanitized_text=original_text,
                metadata=metadata,
            )
        except Exception as exc:
            return self._failure(stage, "malformed_response", error_type=type(exc).__name__)

    def _sanitize(self, text: str, *, stage: str) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(allowed=True, sanitized_text=text, reason="disabled")

        if not isinstance(text, str) or not text:
            return self._failure(stage, "invalid_text", error_type="invalid_input")

        try:
            from google.cloud import modelarmor_v1

            client = self._get_client()
            data_item = modelarmor_v1.DataItem(text=text)
            if stage == "input":
                request = modelarmor_v1.SanitizeUserPromptRequest(
                    name=self.template_name,
                    user_prompt_data=data_item,
                )
                response = client.sanitize_user_prompt(
                    request=request,
                    timeout=self.timeout_seconds,
                )
            else:
                request = modelarmor_v1.SanitizeModelResponseRequest(
                    name=self.template_name,
                    model_response_data=data_item,
                )
                response = client.sanitize_model_response(
                    request=request,
                    timeout=self.timeout_seconds,
                )
            return self._parse_response(response, text, stage)
        except _GuardrailConfigurationError:
            return self._failure(stage, "invalid_configuration")
        except Exception as exc:
            return self._failure(stage, "api_unavailable", error_type=type(exc).__name__)

    def sanitize_user_prompt(self, text: str) -> GuardrailResult:
        """Screen only the latest raw user message before agent/RAG execution."""
        return self._sanitize(text, stage="input")

    def sanitize_model_response(self, text: str) -> GuardrailResult:
        """Screen the complete model response before it reaches the client."""
        return self._sanitize(text, stage="output")

    # Compatibility aliases for the previous partial implementation.
    def sanitize_prompt(self, text: str) -> GuardrailResult:
        return self.sanitize_user_prompt(text)

    def sanitize_response(self, text: str) -> GuardrailResult:
        return self.sanitize_model_response(text)


# Reused by the request lifecycle; no credentials or network call are made at import time.
guardrail = ModelArmorGuardrail()
