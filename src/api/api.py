import requests
import json

# ==== INPUTS from Windmill ====
CHATPROTO_URL = "https://chatproto.pythera.ai/windmill/update"

def main(
    correlation_id: str, llm: str, videos_id: list = [], workspace_path: str = "abc"
):
    payload = {
        "correlation_id": correlation_id,
        "workspace_path": workspace_path,
        "data": {
            "completed": True,
            "result": {
                "text": llm,
                "videos_id": videos_id,
            },
        },
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(
        CHATPROTO_URL, headers=headers, data=json.dumps(payload), timeout=10
    )

    return {
        "status_code": response.status_code,
        "response_text": response.text,
        "sent_payload": payload,
    }
