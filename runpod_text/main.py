# main.py
from fastapi import FastAPI

app = FastAPI()


@app.get("/hello")
async def read_root(prompt: str):
    return {"message": f"{prompt}, 안녕!"}


@app.get("/hello_world")
async def read_item(prompt: str):
    return {"message": f"{prompt}, Hello World!"}
