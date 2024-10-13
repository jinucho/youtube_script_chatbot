import runpod
from fastapi.testclient import TestClient
import asyncio
from main import app
from fastapi.responses import StreamingResponse
import json


client = TestClient(app)


async def runpod_handler(event):
    # RunPod 이벤트에서 요청 데이터를 변환
    endpoint = event["input"].get("endpoint", "/")
    method = event["input"].get("method", "POST").upper()
    request_data = event["input"].get("params", {})
    headers = event["input"].get("headers", {})

    # FastAPI 앱으로 요청을 전달 (TestClient 사용)
    if method == "POST":
        response = client.post(endpoint, json=request_data, headers=headers)
    else:
        response = client.get(endpoint, params=request_data, headers=headers)

    # 응답 처리
    if event["input"].get("endpoint") == "/rag_stream_chat":
        # 스트리밍 응답 처리
        async def stream_generator():
            async for chunk in response.body_iterator:
                yield chunk.decode()

        # FastAPI의 StreamingResponse를 이용하여 스트리밍 응답 생성
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        # 일반 응답 처리
        response_body = await response.body()
        return json.loads(response_body)


def handler(event):
    async def async_handler():
        return await runpod_handler(event)

    if asyncio.get_event_loop().is_running():
        return asyncio.create_task(async_handler())
    else:
        return asyncio.run(async_handler())


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
