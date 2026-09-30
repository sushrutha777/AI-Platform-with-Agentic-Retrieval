"""LiteLLM Gateway for provider-agnostic model access."""

import os
from typing import AsyncGenerator, List, Dict, Any, Optional
import litellm
from app.core.config import settings
from app.core.logging import logger

# Drop litellm internal logs unless debugging
if not settings.DEBUG:
    litellm.suppress_debug_info = True

class LLMGateway:
    """Singleton gateway wrapping LiteLLM Router for resilient provider failover."""

    def __init__(self):
        # Configure primary and fallback models
        self.primary_model = settings.LLM_MODEL
        self.fallback_model = getattr(settings, "FALLBACK_LLM_MODEL", "groq/llama-3.3-70b-versatile")

        # Set keys in environment for LiteLLM to pick up automatically
        if settings.GOOGLE_API_KEY:
            os.environ["GEMINI_API_KEY"] = settings.GOOGLE_API_KEY
            os.environ["VERTEXAI_API_KEY"] = settings.GOOGLE_API_KEY
        if getattr(settings, "OPENAI_API_KEY", None):
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
        if getattr(settings, "ANTHROPIC_API_KEY", None):
            os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY
        if getattr(settings, "GROQ_API_KEY", None):
            os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

        # LangSmith Tracing & Observability
        langsmith_key = getattr(settings, "LANGSMITH_API_KEY", None) or getattr(settings, "LANGCHAIN_API_KEY", None)
        if langsmith_key:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
            os.environ["LANGCHAIN_API_KEY"] = langsmith_key
            proj = getattr(settings, "LANGSMITH_PROJECT", None) or getattr(settings, "LANGCHAIN_PROJECT", None) or "Agentic RAG"
            os.environ["LANGSMITH_PROJECT"] = proj
            os.environ["LANGCHAIN_PROJECT"] = proj
            endpoint = getattr(settings, "LANGSMITH_ENDPOINT", None) or getattr(settings, "LANGCHAIN_ENDPOINT", None) or "https://api.smith.langchain.com"
            os.environ["LANGSMITH_ENDPOINT"] = endpoint
            os.environ["LANGCHAIN_ENDPOINT"] = endpoint
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if not hasattr(litellm, "success_callback") or litellm.success_callback is None:
                litellm.success_callback = []
            if "langsmith" not in litellm.success_callback:
                litellm.success_callback.append("langsmith")

        # Model list for LiteLLM Router
        model_list = [
            {
                "model_name": "primary-model",
                "litellm_params": {
                    "model": self.primary_model,
                    "api_base": settings.LITELLM_API_BASE,
                },
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": self.fallback_model,
                },
            },
        ]

        router_cls = getattr(litellm, "Router", None)
        self.router = (
            router_cls(
                model_list=model_list,
                fallbacks=[{"primary-model": ["fallback-model"]}],
                num_retries=1,  # Short retry on primary (1 retry = 2 total attempts)
                retry_after=0.5,  # Short backoff (0.5s)
                retry_policy="exponential_backoff_retry",
                allowed_fails=1,
                cooldown_time=30,
            )
            if router_cls
            else None
        )

    async def _execute_with_failover(self, messages: List[Dict[str, str]], is_stream: bool = False, **kwargs):
        """Internal helper to execute acompletion with primary model & explicit fallback handling."""
        # Clean kwargs to avoid passing model/api_base/num_retries twice
        kwargs.pop("model", None)
        kwargs.pop("api_base", None)
        kwargs.pop("num_retries", None)

        # Attempt Primary Model
        logger.info(
            f"LLM Gateway Attempting Primary Provider: {self.primary_model} (stream={is_stream})"
        )
        try:
            return await litellm.acompletion(
                model=self.primary_model,
                messages=messages,
                api_base=settings.LITELLM_API_BASE,
                stream=is_stream,
                num_retries=1,
                **kwargs,
            )
        except Exception as primary_err:
            error_type = type(primary_err).__name__
            logger.warning(
                f"LLM Gateway Primary Provider ({self.primary_model}) failed with {error_type}: {primary_err}. Triggering fallback to {self.fallback_model}."
            )

            # Fallback to Groq / Secondary Provider
            try:
                logger.info(
                    f"LLM Gateway Executing Fallback Provider: {self.fallback_model} (stream={is_stream})"
                )
                response = await litellm.acompletion(
                    model=self.fallback_model,
                    messages=messages,
                    stream=is_stream,
                    num_retries=1,
                    **kwargs,
                )
                logger.info(
                    f"LLM Gateway Fallback Provider ({self.fallback_model}) succeeded."
                )
                return response
            except Exception as fallback_err:
                fb_error_type = type(fallback_err).__name__
                logger.error(
                    f"LLM Gateway Fallback Provider ({self.fallback_model}) also failed with {fb_error_type}: {fallback_err}."
                )
                raise RuntimeError("Service is temporarily unavailable. Please try again in a few moments.") from fallback_err

    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream response tokens back with provider failover and mid-stream safety."""
        try:
            response = await self._execute_with_failover(messages, is_stream=True, **kwargs)
            tokens_emitted = False

            try:
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        tokens_emitted = True
                        yield content
            except Exception as stream_err:
                logger.warning(
                    f"Streaming error occurred mid-stream (tokens_emitted={tokens_emitted}): {stream_err}"
                )
                # If mid-stream failure occurs before tokens emitted, retry whole completion with fallback
                if not tokens_emitted:
                    logger.info(
                        f"No tokens were emitted before streaming failure. Triggering full completion fallback to {self.fallback_model}."
                    )
                    fallback_response = await litellm.acompletion(
                        model=self.fallback_model,
                        messages=messages,
                        stream=True,
                        num_retries=1,
                        **kwargs,
                    )
                    async for chunk in fallback_response:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                else:
                    # If tokens were already emitted to client, yield a friendly user notice
                    yield "\n\n[Note: Response stream ended early due to temporary model connection disruption.]"

        except Exception as e:
            logger.error(f"LLM Gateway streaming failed completely: {e}")
            yield "Service is temporarily unavailable. Please try again in a few moments."

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Get a single string completion with provider failover."""
        try:
            response = await self._execute_with_failover(messages, is_stream=False, **kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM Gateway complete failed completely: {e}")
            return "Service is temporarily unavailable. Please try again in a few moments."


# Singleton instance
gateway = LLMGateway()
