# Providers

Every model-backed component (the classification `ModelDetector`, the browser `ChatModel`, and any callbacks you write for adaptive campaigns) uses one client: `browsesafe_pp.OpenAICompatible`. It speaks the OpenAI Chat Completions protocol, so it works with OpenAI, Azure-style gateways, vLLM, SGLang, llama.cpp server, Ollama, LiteLLM, OpenRouter and similar servers.

## Environment variables

| Variable | Meaning | Example |
|---|---|---|
| `BROWSESAFE_MODEL_ENDPOINT` | API base URL. `/chat/completions` is appended unless already present. | `https://api.openai.com/v1`, `http://localhost:8000/v1` |
| `BROWSESAFE_API_KEY` | Bearer token, sent as `Authorization: Bearer ...`. Optional for local servers. | `sk-...` |
| `BROWSESAFE_MODEL` | Default model name. | `gpt-4o-mini`, `Qwen/Qwen2.5-7B-Instruct` |
| `BROWSESAFE_DATASET` | Hugging Face dataset id used by `load()` (not a provider setting, listed for completeness). | `perplexity-ai/browsesafe-plus-plus` |

Arguments override the environment:

```python
import browsesafe_pp as bs

provider = bs.OpenAICompatible(
    endpoint="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-7B-Instruct",
    api_key=None,                              # local server without a key
    timeout=120,
    token_limit_field="max_tokens",            # for servers that do not accept max_completion_tokens
    prices_per_million=(0.15, 0.60),           # optional cost estimate when the server reports none
    extra_body={"top_p": 1.0},                 # any extra request fields
)
```

## Making requests

```python
response = await provider.chat(
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    max_tokens=512, temperature=0,
    tools=None, tool_choice=None, response_format=None, seed=None,
)
response.text          # message content ("" if none)
response.tool_calls    # tuple of ToolCall(id, name, arguments-as-JSON-string)
response.usage         # Usage(model_calls=1, input_tokens, output_tokens, cost_usd, cost_basis)
await provider.aclose()
```

User content may be a string or a list of OpenAI content parts (text and `image_url`), which is how the screenshot view is sent.

## Failure semantics

- No implicit retries. A failed request raises `ProviderError` with `category` (`request_error`, `http_error`, `invalid_response`, `truncated`, `refusal`) and the `usage` it incurred.
- Evaluation turns provider errors into `error` outcomes and unparseable answers into `invalid_prediction` outcomes; both are reported, never silently relabelled.
- `provider.configuration` is safe to record: it includes endpoint, model and settings but never the API key.

## Using another client

Anything with an async `chat(messages, *, max_tokens, temperature, tools, tool_choice)` method that returns an object with `text`, `tool_calls` and `usage` can replace `OpenAICompatible` in `ModelDetector`. For full control, write your own detector function instead (see [Classification](classification.md)).
