"""Shared model and reasoning configuration."""

import os


MODEL = "gpt-6-luna"
MAX_COMPLETION_TOKENS = 4000  # Includes visible output and reasoning tokens.
ALLOWED_REASONING_EFFORTS = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)

_configured_effort = os.environ.get("LUNA_EFFORT", "max").strip().lower()
REASONING_EFFORT = (
    _configured_effort
    if _configured_effort in ALLOWED_REASONING_EFFORTS
    else "max"
)


def _is_reasoning_effort_parameter_error(error):
    """Recognize API errors that specifically reject reasoning_effort."""
    if getattr(error, "param", None) == "reasoning_effort":
        return True

    body = getattr(error, "body", None)
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict) and details.get("param") == "reasoning_effort":
            return True

    message = str(error).casefold()
    return "reasoning_effort" in message and any(
        marker in message
        for marker in (
            "unexpected",
            "unsupported",
            "unknown",
            "unrecognized",
            "invalid",
            "not supported",
            "not allowed",
        )
    )


def create_chat_completion(client, *, reasoning_effort=None, **request):
    """Create a capped completion, retrying once if effort is rejected."""
    effort = REASONING_EFFORT if reasoning_effort is None else reasoning_effort
    if effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            "reasoning_effort must be one of: "
            + ", ".join(sorted(ALLOWED_REASONING_EFFORTS))
        )

    request["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    request["reasoning_effort"] = effort
    try:
        return client.chat.completions.create(**request)
    except Exception as error:
        if not _is_reasoning_effort_parameter_error(error):
            raise
        request.pop("reasoning_effort")
        return client.chat.completions.create(**request)
