import logging
import os
from typing import Dict, Any

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from torch import device
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

load_dotenv()
model = WhisperModel("large-v3", device="cuda")

if device == "cuda":
    logging.info("Using GPU for transcription.")

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
diarization_model = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGING_FACE_TOKEN,
)

def perform_diarization(file_path: str) -> Annotation:
    """Выполняет диаризацию аудиофайла для определения говорящих."""
    try:
        logging.info(f"Performing diarization on file: {file_path}")
        diarization = diarization_model(file_path)
        return diarization
    except Exception as e:
        logging.error(f"Error in perform_diarization: {e}")
        raise


def transcribe_with_diarization(file_path: str, language: str = "kk", task: str = "transcribe") -> Dict[str, Any]:
    """Транскрибирует аудио с диаризацией говорящих."""
    try:
        # Выполняем базовую транскрипцию
        language_arg = None if language == "auto" else language
        result = model.transcribe(
            file_path,
            language=language_arg,
            task=task
        )

        # Обрабатываем результат
        if isinstance(result, tuple) and len(result) == 2:
            segments_generator, info = result
        else:
            segments_generator = result
            info = getattr(result, "info", None)

        transcript_segments = list(segments_generator)
        logging.info(f"Транскрипция успешна: получено {len(transcript_segments)} сегментов")

        # Выполняем диаризацию
        diarization = diarization_model(file_path)

        # Создаем временную шкалу говорящих
        speaker_timeline = {}
        for speaker, track in diarization.itertracks(yield_label=True):
            start_time = track.start
            end_time = track.end
            for i in range(int(start_time * 100), int(end_time * 100)):
                time_point = i / 100.0
                speaker_timeline[time_point] = speaker

        # Сопоставляем сегменты с говорящими
        result_segments = []
        current_speaker = None
        formatted_texts = []
        unique_speakers = set()

        for segment in transcript_segments:
            mid_point = (segment.start + segment.end) / 2
            nearest_time = round(mid_point * 100) / 100

            # Ищем ближайшую временную метку
            if nearest_time not in speaker_timeline:
                for offset in range(1, 50):
                    check_before = nearest_time - offset / 100
                    check_after = nearest_time + offset / 100
                    if check_before in speaker_timeline:
                        speaker_label = speaker_timeline[check_before]
                        break
                    if check_after in speaker_timeline:
                        speaker_label = speaker_timeline[check_after]
                        break
                else:
                    speaker_label = "SPEAKER_1"
            else:
                speaker_label = speaker_timeline[nearest_time]

            unique_speakers.add(speaker_label)

            # Добавляем метку говорящего в текст
            if current_speaker != speaker_label:
                current_speaker = speaker_label
                formatted_texts.append(f"[{speaker_label}]: {segment.text}")
            else:
                formatted_texts.append(segment.text)

            result_segments.append({
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker_label
            })

        # Создаем форматированный текст с указанием говорящих
        formatted_full_text = " ".join(formatted_texts)

        # Обычный текст без форматирования (для обратной совместимости)
        plain_full_text = " ".join(seg["text"] for seg in result_segments)

        # Результат с добавлением списка говорящих
        result_dict = {
            "text": plain_full_text,
            "formatted_text": formatted_full_text,
            "segments": result_segments,
            "speakers": list(unique_speakers)
        }

        # Добавляем метаданные
        if info:
            result_dict["language"] = getattr(info, "language", language)
            result_dict["duration"] = getattr(info, "duration", None)
        else:
            result_dict["language"] = language
            if result_segments:
                result_dict["duration"] = result_segments[-1]["end"]

        return result_dict

    except Exception as e:
        logging.error(f"Ошибка в transcribe_with_diarization: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        logging.info("Откат к транскрипции без диаризации")
        return transcribe_audio(file_path, language, task, enable_diarization=False)

def transcribe_audio(file_path: str, language: str = "kk", task: str = "transcribe",
                     enable_diarization: bool = False) -> Dict[str, Any]:
    """
    Транскрибирует аудио файл с опциональной диаризацией.
    """
    if enable_diarization:
        return transcribe_with_diarization(file_path, language, task)

    try:
        logging.info(f"Transcribing audio file: {file_path}")

        language_arg = None if language == "auto" else language

        # Получаем генератор сегментов
        result = model.transcribe(
            file_path,
            language=language_arg,
            task=task
        )

        # Обрабатываем результат в зависимости от его типа
        if isinstance(result, tuple):
            segments_generator = result[0]
            info = result[1]
        else:
            segments_generator = result
            info = None

        segment_list = list(segments_generator)

        segments_data = []
        texts = []

        for segment in segment_list:
            texts.append(segment.text)
            segments_data.append({
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            })

        transcription = " ".join(texts)

        result_dict = {
            "text": transcription,
            "segments": segments_data
        }

        if info:
            result_dict["language"] = getattr(info, "language", language)
            result_dict["duration"] = getattr(info, "duration", None)

        return result_dict

    except Exception as e:
        logging.error(f"Error in transcribe_audio: {e}")
        raise