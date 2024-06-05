import json
import os
import time
from dataclasses import dataclass
from typing import AsyncIterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from llama_cpp import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

@dataclass
class Model:
    repo_id: str
    filename: str
    chat_format: str
    n_ctx: int

    id: int
    object: str
    created: int
    owned_by: str

debug = os.environ.get('GPTSCRIPT_DEBUG', False) == "true"

model_spec_path = os.environ.get('LLAMA_SPEC_PATH')
if model_spec_path == None:
    print('LLAMA_SPEC_PATH must be defined')
    exit(1)

models = {}
models_cache = {}
try:
    with open(model_spec_path, 'r') as f:
        models_spec = json.loads(f.read())
        for name, m in models_spec.items():
            models[name] = Model(
                repo_id=m['repo_id'], 
                filename=m['filename'], 
                chat_format=m['chat_format'], 
                n_ctx=m['n_ctx'],
                id=name,
                object='assistant',
                created=int(time.time()),
                owned_by='llama.cpp'
            )
except Exception as e:
    print("An error occured while loading the model specifications:", e)
    exit(1)


def log(*args):
    if debug:
        print(*args)


app = FastAPI()

system: str = """
You are task oriented system.
You receive input from a user, process the input from the given instructions, and then output the result.
Your objective is to provide consistent and correct results.
Call the provided tools as needed to complete the task.
You do not need to explain the steps taken, only provide the result to the given instructions.
You are referred to as a tool.
You don't move to the next step until you have a result.
"""


@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()
    log("REQUEST BODY: ", body)
    return await call_next(request)


@app.get("/")
async def get_root():
    return "ok"

@app.get("/v1/models")
async def list_models() -> JSONResponse:
    data: list[dict] = []
    for _, model in models.items():
        data.append({
            "id": model.id,
            "object": model.object,
            "created": model.created,
            "owned_by": model.owned_by
        })
    return JSONResponse(content={"object": "list", "data": data})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.body()
    data = json.loads(data)

    messages = data["messages"]
    messages.insert(0, {"role": "system", "content": system})

    model_name = data['model']
    model = models[model_name]
    if models_cache.get(model_name):
        client = models_cache[model_name]
    else:
        client = Llama.from_pretrained(
          repo_id=model.repo_id,
          filename=model.filename,
          chat_format=model.chat_format,
          tokenizer=LlamaHFTokenizer.from_pretrained(model.repo_id),
          n_ctx=model.n_ctx,
        )
        models_cache[model_name] = client

    stream = client.create_chat_completion(
        model=data["model"],
        messages=messages,
        max_tokens=data.get("max_tokens", None),
        tools=data.get("tools", None),
        tool_choice=data.get("tool_choice", None),
        stream=data.get("stream", False),
        top_p=data.get("top_p", 0.95),
        temperature=data.get("temperature", 0.2),
    )

    async def convert_stream(stream: AsyncStream[ChatCompletionChunk]) -> AsyncIterable[str]:
        for chunk in stream:
            log("CHUNK: ", json.dumps(chunk))
            yield "data: " + json.dumps(chunk) + "\n\n"

    return StreamingResponse(convert_stream(stream), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")),
                log_level="debug" if debug else "critical", reload=debug, access_log=debug)
