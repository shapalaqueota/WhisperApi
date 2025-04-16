import logging
from typing import Dict, Any
from faster_whisper import WhisperModel


model = WhisperModel("large-v3", device="cpu")


def transcribe_audio(file_path: str, language: str = "kk", task: str = "transcribe") -> Dict[str, Any]:

    try:
        # Process with Faster Whisper
        logging.info(f"Transcribing audio file: {file_path}")

        # Set language if provided
        language_arg = None if language == "auto" else language

        # Transcribe the audio file directly
        segments, info = model.transcribe(
            file_path,
            language=language_arg,
            task=task
        )

        # Combine all segments to get the complete transcription
        transcription = " ".join([segment.text for segment in segments])

        result = {
            "text": transcription,
            "language": info.language,
            "duration": info.duration
        }

        return result

    except Exception as e:
        logging.error(f"Error in transcribe_audio: {e}")
        raise