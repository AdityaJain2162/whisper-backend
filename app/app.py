import os
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Body,
    BackgroundTasks,
    Request,
    File,
    UploadFile,
    Form
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import pipeline
from .diarization_pipeline import diarize
import requests
import asyncio
import uuid
import shutil

admin_key = os.environ.get("ADMIN_KEY")
hf_token = os.environ.get("HF_TOKEN")
fly_machine_id = os.environ.get("FLY_MACHINE_ID")

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device="cuda:0",
    model_kwargs={"attn_implementation": "flash_attention_2"},
)

app = FastAPI()
loop = asyncio.get_event_loop()
running_tasks = {}

class WebhookBody(BaseModel):
    url: str
    header: dict[str, str] = {}

def process(
    file_path_or_url: str,
    task: str,
    language: str,
    batch_size: int,
    timestamp: str,
    diarise_audio: bool,
    webhook: WebhookBody | None = None,
    task_id: str | None = None,
    is_file: bool = True,
):
    errorMessage: str | None = None
    outputs = {}
    try:
        generate_kwargs = {
            "task": task,
            "language": None if language == "None" else language,
        }

        # Handle processing based on whether it's a file or a URL
        if is_file:
            outputs = pipe(
                file_path_or_url,
                chunk_length_s=30,
                batch_size=batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps="word" if timestamp == "word" else True,
            )
        else:
            outputs = pipe(
                file_path_or_url,
                chunk_length_s=30,
                batch_size=batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps="word" if timestamp == "word" else True,
            )

        if diarise_audio:
            speakers_transcript = diarize(
                hf_token,
                file_path_or_url,
                outputs,
            )
            outputs["speakers"] = speakers_transcript
    except asyncio.CancelledError:
        errorMessage = "Task Cancelled"
    except Exception as e:
        errorMessage = str(e)

    if task_id is not None:
        del running_tasks[task_id]

    if webhook is not None:
        webhookResp = (
            {"output": outputs, "status": "completed", "task_id": task_id}
            if errorMessage is None
            else {"error": errorMessage, "status": "error", "task_id": task_id}
        )

        if fly_machine_id is not None:
            webhookResp["fly_machine_id"] = fly_machine_id

        requests.post(
            webhook.url,
            headers=webhook.header,
            json=(webhookResp),
        )

    if errorMessage is not None:
        raise Exception(errorMessage)

    return outputs

@app.post("/process_file/")
async def process_file(
    file: UploadFile = File(...),
    task: str = Form(default="transcribe"),
    language: str = Form(default="None"),
    batch_size: int = Form(default=64),
    timestamp: str = Form(default="chunk"),
    diarise_audio: bool = Form(default=False),
    is_async: bool = Form(default=False),
    managed_task_id: str | None = Form(default=None),
    webhook_url: str | None = Form(default=None)
):
    if diarise_audio and hf_token is None:
        raise HTTPException(status_code=500, detail="Missing Hugging Face Token")

    if is_async and webhook_url is None:
        raise HTTPException(status_code=400, detail="Webhook is required for async tasks")

    task_id = managed_task_id if managed_task_id else str(uuid.uuid4())
    temp_file_path = f"/tmp/{task_id}_{file.filename}"

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        resp = {}
        if is_async:
            backgroundTask = asyncio.ensure_future(
                loop.run_in_executor(
                    None,
                    process,
                    temp_file_path,
                    task,
                    language,
                    batch_size,
                    timestamp,
                    diarise_audio,
                    webhook_url,
                    task_id,
                    True  # is_file
                )
            )
            running_tasks[task_id] = backgroundTask
            resp = {
                "detail": "Task is being processed in the background",
                "status": "processing",
                "task_id": task_id,
            }
        else:
            running_tasks[task_id] = None
            outputs = process(
                temp_file_path,
                task,
                language,
                batch_size,
                timestamp,
                diarise_audio,
                webhook_url,
                task_id,
                True  # is_file
            )
            resp = {
                "output": outputs,
                "status": "completed",
                "task_id": task_id,
            }
        if fly_machine_id:
            resp["fly_machine_id"] = fly_machine_id
        return resp
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_file_path)

@app.post("/process_url/")
def process_url(
    url: str = Body(),
    task: str = Body(default="transcribe", enum=["transcribe", "translate"]),
    language: str = Body(default="None"),
    batch_size: int = Body(default=64),
    timestamp: str = Body(default="chunk", enum=["chunk", "word"]),
    diarise_audio: bool = Body(default=False),
    webhook: WebhookBody | None = None,
    is_async: bool = Body(default=False),
    managed_task_id: str | None = Body(default=None),
):
    if not url.lower().startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")

    if diarise_audio and hf_token is None:
        raise HTTPException(status_code=500, detail="Missing Hugging Face Token")

    if is_async and webhook is None:
        raise HTTPException(
            status_code=400, detail="Webhook is required for async tasks"
        )

    task_id = managed_task_id if managed_task_id else str(uuid.uuid4())

    try:
        resp = {}
        if is_async:
            backgroundTask = asyncio.ensure_future(
                loop.run_in_executor(
                    None,
                    process,
                    url,
                    task,
                    language,
                    batch_size,
                    timestamp,
                    diarise_audio,
                    webhook,
                    task_id,
                    False  # is_file
                )
            )
            running_tasks[task_id] = backgroundTask
            resp = {
                "detail": "Task is being processed in the background",
                "status": "processing",
                "task_id": task_id,
            }
        else:
            running_tasks[task_id] = None
            outputs = process(
                url,
                task,
                language,
                batch_size,
                timestamp,
                diarise_audio,
                webhook,
                task_id,
                False  # is_file
            )
            resp = {
                "output": outputs,
                "status": "completed",
                "task_id": task_id,
            }
        if fly_machine_id:
            resp["fly_machine_id"] = fly_machine_id
        return resp
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks")
def tasks():
    return {"tasks": list(running_tasks.keys())}

@app.get("/status/{task_id}")
def status(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]

    if task is None:
        return {"status": "processing"}
    elif task.done() is False:
        return {"status": "processing"}
    else:
        return {"status": "completed", "output": task.result()}

@app.delete("/cancel/{task_id}")
def cancel(task_id: str):
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = running_tasks[task_id]
    if task is None:
        return HTTPException(status_code=400, detail="Not a background task")
    elif task.done() is False:
        task.cancel()
        del running_tasks[task_id]
        return {"status": "cancelled"}
    else:
        return {"status": "completed", "output": task.result()}
