# runpod_wrapper.py
import os
import runpod
import requests

RUNPOD_INTERNAL_API_HOST = os.getenv(
    "RUNPOD_INTERNAL_API_HOST", "http://localhost:8000"
)


def handler(event):
    print(event)
    input_data = event.get("input", {})
    endpoint = input_data.get("endpoint", "").lstrip(
        "/"
    )  # Remove leading slash if present
    params = input_data.get("params", {})
    prompt = params.get("prompt", "")

    if endpoint in ["hello", "hello_world"]:
        response = requests.get(
            f"{RUNPOD_INTERNAL_API_HOST}/{endpoint}", params={"prompt": prompt}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API request failed with status code {response.status_code}"
            }
    else:
        return {"error": "Invalid endpoint"}


runpod.serverless.start({"handler": handler})
