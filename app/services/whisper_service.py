import whisper
from app.core.config import settings
import numpy as np
import torch

import sys

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


if torch.cuda.is_available():
    print("GPU")
    device = "cuda"
else:
    print("CPU")
    device = "cpu"

model = whisper.load_model(settings.whisper_model).to(device)


async def transcribe_audio_with_progress(y: np.ndarray, sr: int, websocket=None) -> str:
    audio = whisper.pad_or_trim(y)

    if websocket:
        await websocket.send_text("Starting transcription...")

    result = model.transcribe(audio, language=settings.whisper_language, verbose=True)
    print(result)

    if isinstance(result, dict) and "text" in result:
        transcription = result["text"]



        if websocket:
            await websocket.send_text("Transcription completed.")
            await websocket.send_text(transcription)

        return transcription
    else:
        if websocket:
            await websocket.send_text("Transcription failed.")
        raise ValueError("Unexpected response format from Whisper model")
