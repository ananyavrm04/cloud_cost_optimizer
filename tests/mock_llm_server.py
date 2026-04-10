from fastapi import FastAPI


app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(payload: dict) -> dict:
    # Deterministic mock response for dockerized E2E checks.
    return {
        "id": "chatcmpl-mock-1",
        "object": "chat.completion",
        "created": 0,
        "model": payload.get("model", "mock-model"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"action_type":"skip","resource_id":"","new_size":"","new_pricing":""}'
                    ),
                },
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
