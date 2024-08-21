import whisper
from app.core.config import settings
import numpy as np
import torch

if torch.cuda.is_available():
    print("GPU")
    device = "cuda"
else:
    print("CPU")
    device = "cpu"
model = whisper.load_model(settings.whisper_model).to(device)


async def transcribe_audio(y: np.ndarray, sr: int) -> str:
    audio = whisper.pad_or_trim(y)

    result = model.transcribe(audio, language=settings.whisper_language)

    if isinstance(result, dict) and "text" in result:
        return result["text"]
    else:
        raise ValueError("Unexpected response format from Whisper model")
