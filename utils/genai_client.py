import google.generativeai as genai
import requests
from dotenv import load_dotenv

from utils.config import config
from utils.logging_utils import get_logger

load_dotenv()

# Initialize clients based on provider
logger = get_logger("GenAIClient")

def _get_provider_and_model():
    """Resolve provider and model based on configuration.

    Priority:
    1. If `LLM_MODEL` contains the pattern `<provider>:<model>` → honour it and override
       whatever is in `LLM_PROVIDER`.
    2. Otherwise use `LLM_PROVIDER` as the provider and `LLM_MODEL` as the model string
       exactly as written.  This lets users write:

           LLM_PROVIDER: "gemini"
           LLM_MODEL:    "gemini-2.5-flash"

       or

           LLM_PROVIDER: "anthropic"
           LLM_MODEL:    "claude-sonnet-4"

    3. If `LLM_PROVIDER` is missing or set to "auto", fall back to simple heuristics
       based on model name prefixes (gemini / claude).
    """

    raw_provider = config.llm_provider.lower().strip()
    model_config = config.llm_model.strip()

    # --- 1. provider:model explicit syntax ---------------------------------
    if ":" in model_config:
        provider, model = model_config.split(":", 1)
        return provider.lower(), model

    # --- 2. explicit provider in YAML --------------------------------------
    if raw_provider not in {"auto", ""}:
        return raw_provider, model_config

    # --- 3. auto-detect provider from model name ---------------------------
    if model_config.startswith("gemini"):
        return "gemini", model_config
    if model_config.startswith("claude"):
        return "anthropic", model_config
    if model_config.startswith("gpt") or model_config.startswith("o"):
        return "openai", model_config

    # Fallback to openrouter
    return "openrouter", model_config

def _init_clients():
    """Initialize the appropriate client(s)"""
    provider, model = _get_provider_and_model()
    
    clients = {}
    
    if provider == "gemini":
        try:
            genai.configure(api_key=config.google_api_key)
            clients["gemini"] = genai
            logger.debug("Configured Google Gemini client with model %s", model)
        except Exception as e:
            logger.warning("Failed to initialize Gemini client: %s", e)
    
    elif provider == "openrouter":
        try:
            clients["openrouter"] = {
                "api_key": config.openrouter_api_key,
                "base_url": "https://openrouter.ai/api/v1",
            }
            logger.debug("Configured OpenRouter client with model %s", model)
        except Exception as e:
            logger.warning("Failed to initialise OpenRouter client: %s", e)
    
    elif provider == "claude":  # legacy alias routed via OpenRouter
        try:
            clients["claude"] = {
                "api_key": config.openrouter_api_key,
                "base_url": "https://openrouter.ai/api/v1",
            }
            logger.debug("Configured OpenRouter (legacy 'claude') client with model %s", model)
        except Exception as e:
            logger.warning("Failed to initialise OpenRouter (legacy) client: %s", e)
    
    elif provider == "anthropic":
        try:
            clients["anthropic"] = {
                "api_key": config.anthropic_api_key,
                "base_url": "https://api.anthropic.com/v1",
                "api_version": "2023-06-01",
            }
            logger.debug("Configured direct Anthropic client with model %s", model)
        except Exception as e:
            logger.warning("Failed to initialize Anthropic client: %s", e)
    
    elif provider == "openai":
        try:
            clients["openai"] = {
                "api_key": config.openai_api_key,
                "base_url": "https://api.openai.com/v1",
            }
            logger.debug("Configured OpenAI client with model %s", model)
        except Exception as e:
            logger.warning("Failed to initialise OpenAI client: %s", e)
    
    else:
        logger.error("Unknown provider: %s. Supported providers: gemini, openrouter, anthropic, openai", provider)
        raise ValueError(f"Unknown provider: {provider}")
    
    return clients, provider, model

# Initialize clients
_clients, _current_provider, _current_model = _init_clients()

# Log the initialized configuration
logger.info("🤖 LLM Client initialized - Provider: %s, Model: %s", _current_provider.upper(), _current_model)
print(f"🤖 LLM Client ready: {_current_provider.upper()}:{_current_model}")


def _chat_with_gemini(messages: list[dict], temperature: float, model: str) -> str:
    """Handle chat with Gemini"""
    gemini_client = _clients["gemini"]
    model_instance = gemini_client.GenerativeModel(model)
    full_prompt = "\n".join(m["content"] for m in messages)
    logger.debug("Gemini prompt length=%d", len(full_prompt))

    resp = model_instance.generate_content(
        full_prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": config.llm_max_tokens,
        },
    )

    # Check if response has valid parts before accessing text
    if not resp.parts:
        finish_reason = (
            resp.candidates[0].finish_reason if resp.candidates else "unknown"
        )
        logger.error(
            "No valid response parts returned. Finish reason: %s", finish_reason
        )

        # Handle different finish reasons
        if finish_reason == 2:  # SAFETY filter hit
            logger.warning(
                "Gemini safety filter triggered. Returning empty response so pipeline can continue."
            )
            return ""  # graceful degradation
        elif finish_reason == 3:  # RECITATION
            logger.warning(
                "Gemini recitation filter triggered. Returning empty response."
            )
            return ""
        elif finish_reason == 4:  # OTHER
            logger.warning(
                "Gemini returned OTHER finish reason. Returning empty response."
            )
            return ""
        else:
            logger.warning(
                "Gemini failed with finish reason %s. Returning empty response.",
                finish_reason,
            )
            return ""

    logger.info("✅ Gemini response received (%d chars)", len(resp.text))
    logger.debug("Gemini response details - length: %d", len(resp.text))
    return resp.text.strip()


def _chat_with_openrouter(messages: list[dict], temperature: float, model: str) -> str:
    """Generic helper for chatting with any model exposed through OpenRouter."""

    # The key in `_clients` will be either "openrouter" (preferred) or "claude" (legacy)
    provider_key = "openrouter" if "openrouter" in _clients else "claude"
    or_client = _clients[provider_key]
    
    # Convert messages to OpenAI-compatible format
    openai_messages = []
    
    for msg in messages:
        if msg.get("role") in ["system", "user", "assistant"]:
            openai_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        else:
            # If no role specified, treat as user message
            openai_messages.append({
                "role": "user", 
                "content": msg["content"]
            })
    
    # If no proper messages, create one from all content
    if not openai_messages:
        full_content = "\n".join(m["content"] for m in messages)
        openai_messages = [{"role": "user", "content": full_content}]
    
    logger.debug("Converted to %d OpenRouter messages", len(openai_messages))
    
    # Prepare OpenRouter API request
    headers = {
        "Authorization": f"Bearer {or_client['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",  # Optional: helps with rate limits
        "X-Title": "Kernel Development System"  # Optional: identifies your app
    }
    
    payload = {
        "model": model,
        "messages": openai_messages,
        "temperature": temperature,
        "max_tokens": config.llm_max_tokens,
    }
    
    try:
        response = requests.post(
            f"{or_client['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text from OpenAI-compatible response
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                text_content = choice["message"]["content"]
                logger.info("✅ OpenRouter response received (%d chars)", len(text_content))
                logger.debug("OpenRouter response details - length: %d", len(text_content))
                return text_content.strip()
        
        logger.warning("OpenRouter returned empty response")
        return ""
        
    except requests.exceptions.RequestException as e:
        logger.error("OpenRouter API request failed: %s", e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                logger.error("OpenRouter error details: %s", error_detail)
            except:
                logger.error("OpenRouter error response: %s", e.response.text)
        raise RuntimeError(f"OpenRouter API error: {e}")
    
    except Exception as e:
        logger.error("Unexpected error with OpenRouter API: %s", e)
        raise


def _chat_with_anthropic(messages: list[dict], temperature: float, model: str) -> str:
    """Call Anthropic Claude directly via v1/messages endpoint."""

    anthropic_client = _clients["anthropic"]

    # Anthropic Messages API expects:
    #   • Optional top-level `system` string
    #   • `messages` array with roles user/assistant (no system)
    system_prompt = None
    processed_msgs = []

    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            if system_prompt is None:  # take the first system message only
                system_prompt = m["content"]
            # Skip adding to message list – not allowed in Anthropic
            continue
        # Only pass through roles allowed by Anthropic (user/assistant)
        if role not in {"user", "assistant"}:
            role = "user"
        processed_msgs.append({"role": role, "content": m["content"]})

    payload = {
        "model": model,
        "messages": processed_msgs,
        "temperature": temperature,
        "max_tokens": config.llm_max_tokens,
    }
    if system_prompt is not None:
        payload["system"] = system_prompt

    try:
        response = requests.post(
            f"{anthropic_client['base_url']}/messages",
            headers={
                "x-api-key": anthropic_client["api_key"],
                "anthropic-version": anthropic_client["api_version"],
                "content-type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()

        # Anthropic returns a list of content blocks; concatenate all text parts.
        if "content" in result and isinstance(result["content"], list):
            parts = [part.get("text", "") for part in result["content"] if part.get("type") == "text"]
            text_content = "".join(parts)
        else:
            text_content = result.get("content", "")

        if not text_content:
            logger.warning("Anthropic returned empty response")
            return ""

        logger.info("✅ Anthropic response received (%d chars)", len(text_content))
        logger.debug("Anthropic response details - length: %d", len(text_content))
        return text_content.strip()

    except requests.exceptions.RequestException as e:
        logger.error("Anthropic API request failed: %s", e)
        if hasattr(e, "response") and e.response is not None:
            try:
                logger.error("Anthropic error response: %s", e.response.text)
            except Exception:
                pass
        raise RuntimeError(f"Anthropic API error: {e}")

    except Exception as e:
        logger.error("Unexpected error with Anthropic API: %s", e)
        raise


def _chat_with_openai(messages: list[dict], temperature: float, model: str) -> str:
    """Chat completion using the official OpenAI API."""

    oa_client = _clients["openai"]

    # Sanitize / convert messages to OpenAI format (system/user/assistant)
    oa_messages: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role not in {"system", "user", "assistant"}:
            role = "user"
        oa_messages.append({"role": role, "content": msg["content"]})

    if not oa_messages:
        full_content = "\n".join(m["content"] for m in messages)
        oa_messages = [{"role": "user", "content": full_content}]

    # Decide which endpoint to use.
    # If this is a simple single-turn prompt (e.g. one user message) and the
    # model name contains a period (e.g. "gpt-4.1") then prefer the newer
    # /responses endpoint which is recommended by OpenAI for these models.

    use_responses_api = (
        len(oa_messages) == 1
        and oa_messages[0]["role"] == "user"
        and "." in model  # heuristic for new numbered snapshots like gpt-4.1
    )

    headers = {
        "Authorization": f"Bearer {oa_client['api_key']}",
        "Content-Type": "application/json",
    }

    try:
        if use_responses_api:
            resp_payload = {
                "model": model,
                "input": oa_messages[0]["content"],
                "temperature": temperature,
                "max_output_tokens": config.llm_max_tokens,
            }

            response = requests.post(
                f"{oa_client['base_url']}/responses",
                headers=headers,
                json=resp_payload,
                timeout=60,
            )

            # If the endpoint isn't found (404) or method not allowed, fall back.
            if response.status_code in {404, 405}:
                logger.debug("/responses endpoint unavailable; falling back to /chat/completions")
                use_responses_api = False  # toggle fall-through
            else:
                response.raise_for_status()
                result = response.json()

                # Extract aggregated text from the response object
                try:
                    texts = []
                    for item in result.get("output", []):
                        if item.get("type") == "message":
                            for cpart in item.get("content", []):
                                if cpart.get("type") in {"output_text", "text"}:
                                    texts.append(cpart.get("text", ""))
                    combined = "\n".join(texts).strip()
                except Exception:
                    combined = ""

                if combined:
                    logger.info("✅ OpenAI /responses reply received (%d chars)", len(combined))
                    return combined
                logger.warning("OpenAI /responses returned empty output – will fallback to /chat/completions")
                use_responses_api = False  # fallback if empty

        if not use_responses_api:
            # --- existing chat/completions logic (mostly unchanged) ---
            payload = {
                "model": model,
                "messages": oa_messages,
                "temperature": temperature,
                "max_tokens": config.llm_max_tokens,
            }

            for _attempt in range(3):
                response = requests.post(
                    f"{oa_client['base_url']}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 400:
                    try:
                        err = response.json().get("error", {})

                        if (
                            err.get("code") == "unsupported_parameter"
                            and "max_tokens" in err.get("message", "")
                            and "max_completion_tokens" in err.get("message", "")
                            and "max_completion_tokens" not in payload
                        ):
                            logger.info("OpenAI model rejected 'max_tokens'; retrying with 'max_completion_tokens'.")
                            payload.pop("max_tokens", None)
                            payload["max_completion_tokens"] = config.llm_max_tokens
                            continue

                        if (
                            err.get("code") == "unsupported_value"
                            and err.get("param") == "temperature"
                            and payload.get("temperature", 1) != 1
                        ):
                            logger.info("OpenAI model rejected custom temperature; retrying with default 1.")
                            payload["temperature"] = 1
                            continue
                    except ValueError:
                        pass

                break  # success or non-retryable error

            response.raise_for_status()
            result = response.json()

            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                content = (
                    choice.get("message", {}).get("content")
                    if isinstance(choice.get("message"), dict)
                    else None
                )
                if content:
                    logger.info("✅ OpenAI chat response received (%d chars)", len(content))
                    return content.strip()

            logger.warning("OpenAI chat/completions returned empty response")
            return ""

    except requests.exceptions.RequestException as e:
        logger.error("OpenAI API request failed: %s", e)
        if hasattr(e, "response") and e.response is not None:
            logger.error("OpenAI error response: %s", e.response.text)
        raise RuntimeError(f"OpenAI API error: {e}")

    except Exception as e:
        logger.error("Unexpected error with OpenAI API: %s", e)
        raise


def chat(messages: list[dict], temperature: float = None) -> str:
    """
    Generate chat completion using the configured LLM provider.
    
    Args:
        messages: List of message dictionaries with 'content' and optionally 'role'
        temperature: Temperature for generation (uses config default if None)
    
    Returns:
        Generated text response
    """
    # Use configured temperature if not provided
    if temperature is None:
        temperature = config.llm_temperature

    provider, model = _get_provider_and_model()
    
    # Log which model is being used (INFO level for visibility)
    logger.info("🚀 Generating response with %s:%s (temp=%.2f)", 
                provider.upper(), model, temperature)
    
    logger.debug(
        "Chat details - %d messages, provider: %s, model: %s",
        len(messages), provider, model,
    )

    try:
        if provider == "gemini":
            if "gemini" not in _clients:
                error_msg = "❌ Gemini client not initialized. Check GOOGLE_API_KEY."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            return _chat_with_gemini(messages, temperature, model)
        
        elif provider in {"claude", "openrouter"}:
            key = provider if provider in _clients else ("openrouter" if "openrouter" in _clients else "claude")
            if key not in _clients:
                error_msg = "❌ OpenRouter client not initialized. Check OPENROUTER_API_KEY."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            return _chat_with_openrouter(messages, temperature, model)
        
        elif provider == "anthropic":
            if "anthropic" not in _clients:
                error_msg = "❌ Anthropic client not initialized. Check ANTHROPIC_API_KEY."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            return _chat_with_anthropic(messages, temperature, model)
        
        elif provider == "openai":
            if "openai" not in _clients:
                error_msg = "❌ OpenAI client not initialized. Check OPENAI_API_KEY."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            return _chat_with_openai(messages, temperature, model)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")

    except requests.exceptions.RequestException as e:
        logger.error("OpenRouter API error: %s", e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                if "content_policy" in str(error_detail).lower() or "safety" in str(error_detail).lower():
                    logger.warning("Claude content policy triggered. Returning empty response.")
                    return ""
            except:
                pass
        raise RuntimeError(f"OpenRouter API error: {e}")
    
    except Exception as e:
        if "response.text" in str(e) or "valid `Part`" in str(e):
            # Handle Gemini-specific errors
            logger.error("Gemini response validation failed: %s", e)
            raise RuntimeError(
                "The AI model's response was filtered or blocked. This may be due to content policy restrictions. Please try rephrasing your request."
            )
        else:
            logger.error("Unexpected error with LLM API: %s", e)
            raise


def get_current_provider() -> str:
    """Get the currently configured provider"""
    provider, _ = _get_provider_and_model()
    return provider


def get_current_model() -> str:
    """Get the currently configured model"""
    _, model = _get_provider_and_model()
    return model


def get_available_providers() -> list[str]:
    """Get list of available providers based on API keys"""
    available = []
    
    try:
        config.google_api_key
        available.append("gemini")
    except RuntimeError:
        pass
    
    # OpenRouter availability
    try:
        config.openrouter_api_key
        if "openrouter" not in available:
            available.append("openrouter")
    except RuntimeError:
        pass

    # Direct Anthropic availability (can share same key)
    try:
        config.anthropic_api_key
        if "anthropic" not in available:
            available.append("anthropic")
    except RuntimeError:
        pass
    
    # OpenAI availability
    try:
        config.openai_api_key
        if "openai" not in available:
            available.append("openai")
    except RuntimeError:
        pass
    
    return available



def test_provider_connection(provider: str = None) -> dict:
    """Test connection to the specified provider or current provider"""
    if provider is None:
        provider, _ = _get_provider_and_model()
    
    provider = provider.lower()
    result = {"provider": provider, "available": False, "error": None}
    
    try:
        if provider == "gemini":
            config.google_api_key  # This will raise if not set
            result["available"] = True
            result["message"] = "Gemini API key is configured"
        elif provider == "openrouter":
            config.openrouter_api_key
            result["available"] = True
            result["message"] = "OpenRouter API key is configured"
        elif provider == "anthropic":
            config.anthropic_api_key
            result["available"] = True
            result["message"] = "Anthropic API key is configured"
        elif provider == "claude":  # legacy alias
            config.openrouter_api_key
            result["available"] = True
            result["message"] = "OpenRouter API key is configured (legacy alias)"
        elif provider == "openai":
            config.openai_api_key
            result["available"] = True
            result["message"] = "OpenAI API key is configured"
        else:
            result["error"] = f"Unknown provider: {provider}"
    except RuntimeError as e:
        result["error"] = str(e)
    
    return result