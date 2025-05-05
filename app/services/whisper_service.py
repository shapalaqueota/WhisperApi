# import os
# from dotenv import load_dotenv
# import time
# import librosa
# import torch
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline as hf_pipeline
# from pyannote.audio import Pipeline as PyannotePipeline
# from openai import OpenAI
# from typing import Optional, Dict, Any, List

# # ----------------------
# # Загрузка .env и конфигурация
# # ----------------------
# load_dotenv()
# MODEL_ID        = os.getenv("WHISPER_MODEL_ID", "nocturneFlow/whisper-kk-diploma")
# HF_TOKEN        = os.getenv("HF_TOKEN")
# OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
# GPT_MODEL       = os.getenv("GPT_MODEL", "gpt-4o")  # или другой поддерживаемый
# DEVICE_NAME     = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE          = 0 if DEVICE_NAME == "cuda" else -1

# # Инициализация OpenAI SDK v1
# if not OPENAI_API_KEY:
#     print("[WARNING] OPENAI_API_KEY not set. GPT enhancement unavailable.")
#     openai_client = None
# else:
#     openai_client = OpenAI(api_key=OPENAI_API_KEY)
#     print("[INFO] OpenAI client initialized.")

# # ----------------------
# # Загрузка моделей Whisper
# # ----------------------
# processor = WhisperProcessor.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
# model     = WhisperForConditionalGeneration.from_pretrained(
#     MODEL_ID, use_auth_token=HF_TOKEN
# ).to(DEVICE_NAME)

# # Полная транскрипция (pipeline)
# asr_pipeline = hf_pipeline(
#     task="automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     device=DEVICE,
#     chunk_length_s=30,
#     stride_length_s=(5, 5),
# )

# # Диаризатор
# diarizer = PyannotePipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1",
#     use_auth_token=HF_TOKEN,
# )

# # Параметры сегментации
# MIN_SEGMENT_LEN = 0.5
# MERGE_GAP       = 0.2


# def enhance_text_with_gpt(text: str, language: str, prompt_template: Optional[str] = None) -> str:
#     """
#     Улучшает текст через GPT-4o с помощью OpenAI SDK v1.
#     Возвращает оригинал при отсутствии клиента или ошибке.
#     """
#     if not openai_client:
#         print("[INFO] GPT enhancement skipped: client unavailable.")
#         return text
#     try:
#         # Шаблон промпта
#         if not prompt_template:
#             prompt_template = (
#                 "Мына транскрипцияны тексерiп, қателерді түзетіңіз және форматтаңыз: {text}"
#                 if language == "kk"
#                 else "Check and correct this transcription, fix grammar and formatting: {text}"
#             )
#         prompt = prompt_template.format(text=text)
#         print(f"[DEBUG] Sending prompt (~{len(text)} chars) to GPT v1 api")
#         # Вызов нового клиента
#         response = openai_client.chat.completions.create(
#             model=GPT_MODEL,
#             messages=[
#                 {"role": "system",  "content": f"You are a helpful assistant improving {language} transcriptions."},
#                 {"role": "user",    "content": prompt}
#             ],
#             temperature=0.3,
#             max_tokens=2048
#         )
#         enhanced = response.choices[0].message.content.strip()
#         print("[DEBUG] GPT enhancement received.")
#         return enhanced
#     except Exception as e:
#         print(f"[ERROR] GPT enhancement failed: {e}")
#         return text


# def enhance_segments_with_gpt(segments: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
#     """Применяет GPT к каждому сегменту."""
#     if not segments:
#         return segments
#     enhanced = []
#     for seg in segments:
#         seg_copy = seg.copy()
#         seg_copy['text_before_gpt'] = seg_copy['text']
#         seg_copy['text'] = enhance_text_with_gpt(seg_copy['text'], language)
#         enhanced.append(seg_copy)
#     return enhanced


# def transcribe_audio(
#     audio_path: str,
#     language: str = "kk",
#     task: str = "transcribe",
#     enable_diarization: bool = True,
#     gpt_prompt: Optional[str] = None
# ) -> dict:
#     """
#     ASR + обязательная GPT постобработка с OpenAI SDK v1.
#     """
#     t0 = time.time()
#     mode = task if task in ("transcribe", "translate") else "transcribe"

#     # Диаризация и сегментация
#     if enable_diarization:
#         ann = diarizer({"audio": audio_path})
#         raw = [{"start": round(turn.start,2), "end": round(turn.end,2), "speaker": spk}
#                for turn, _, spk in ann.itertracks(yield_label=True)]
#         filt = [s for s in raw if s['end'] - s['start'] >= MIN_SEGMENT_LEN]
#         merged = []
#         for seg in sorted(filt, key=lambda x: x['start']):
#             if merged and seg['speaker']==merged[-1]['speaker'] and seg['start']-merged[-1]['end']<=MERGE_GAP:
#                 merged[-1]['end'] = seg['end']
#             else:
#                 merged.append(seg.copy())
#         segments = []
#         lines = []
#         speakers = []
#         for seg in merged:
#             st, ed, spk = seg['start'], seg['end'], seg['speaker']
#             if spk not in speakers:
#                 speakers.append(spk)
#             speech, sr = librosa.load(audio_path, offset=st, duration=ed-st, sr=16000)
#             inp = processor(speech, sampling_rate=sr, return_tensors='pt').input_features.to(DEVICE_NAME)
#             kwargs = {'task':'translate'} if mode=='translate' else {}
#             ids = model.generate(inp, **kwargs)
#             txt = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
#             segments.append({'start':st,'end':ed,'speaker':spk,'text':txt})
#             lines.append(f"{spk}: {txt}")
#         full_text = ' '.join([s['text'] for s in segments])
#         formatted = "\n".join(lines)
#     else:
#         kwargs = {'task':'translate'} if mode=='translate' else {}
#         res = asr_pipeline(audio_path, **kwargs)
#         full_text = res.get('text','').strip() if isinstance(res,dict) else str(res)
#         segments, formatted, speakers = None, full_text, []

#     # До постобработки
#     print('==== BEFORE GPT ENHANCEMENT ====')
#     print(formatted)
#     print('==== END BEFORE ====')

#     # Постобработка GPT
#     enhanced_text = enhance_text_with_gpt(full_text, language, gpt_prompt)
#     enhanced_segments = enhance_segments_with_gpt(segments, language) if segments else None
#     enhanced_lines = ("\n".join([f"{s['speaker']}: {s['text']}" for s in enhanced_segments])
#                       if enhanced_segments else enhanced_text)

#     # После постобработки
#     print('==== AFTER GPT ENHANCEMENT ====')
#     print(enhanced_lines)
#     print('==== END AFTER ====')

#     duration = round(time.time()-t0,2)
#     return {
#         'text': full_text,
#         'segments': segments,
#         'formatted_text': formatted,
#         'speakers': speakers,
#         'duration': duration,
#         'language': language,
#         'task': mode,
#         'enhanced_text': enhanced_text,
#         'enhanced_segments': enhanced_segments,
#         'enhanced_formatted_text': enhanced_lines,
#         'enhancement_duration': round(time.time()-duration,2)
#     }

import librosa, torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline as hf_pipeline
import os

# загрузка модели Whisper (как у вас сейчас)
processor = WhisperProcessor.from_pretrained(os.getenv("WHISPER_MODEL_PATH"), use_auth_token=os.getenv("HF_TOKEN"))
model = WhisperForConditionalGeneration.from_pretrained(os.getenv("WHISPER_MODEL_PATH"), use_auth_token=os.getenv("HF_TOKEN")).to(os.getenv("WHISPER_DEVICE"))

asr_pipeline = hf_pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=int(os.getenv("DEVICE")) if torch.cuda.is_available() else -1,
    chunk_length_s=30,
    stride_length_s=(5,5),
)

def transcribe_segment(audio_path: str, start: float, end: float, task="transcribe"):
    speech, sr = librosa.load(audio_path, offset=start, duration=end-start, sr=16000)
    inp = processor(speech, sampling_rate=sr, return_tensors="pt").input_features.to(os.getenv("WHISPER_DEVICE"))
    kwargs = {'task': 'translate'} if task=='translate' else {}
    ids = model.generate(inp, **kwargs)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def transcribe_full(audio_path: str, task="transcribe"):
    kwargs = {'task':'translate'} if task=='translate' else {}
    res = asr_pipeline(audio_path, **kwargs)
    return res['text'] if isinstance(res, dict) else str(res)
