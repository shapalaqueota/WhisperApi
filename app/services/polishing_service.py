# from openai import OpenAI
# import os

# _openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")

# def polish_text(text: str, language: str, prompt_template: str=None) -> str:
#     if not _openai:
#         return text
#     if not prompt_template:
#         prompt_template = (
#             "Мына транскрипцияны тексерiп, қателерді түзетіңіз және форматтаңыз: {text}"
#             if language=="kk"
#             else "Check and correct this transcription, fix grammar and formatting: {text}"
#         )
#     prompt = prompt_template.format(text=text)
#     resp = _openai.chat.completions.create(
#         model=GPT_MODEL,
#         messages=[
#             {"role":"system",  "content":f"You are a helpful assistant improving {language} transcriptions."},
#             {"role":"user",    "content":prompt}
#         ],
#         temperature=0.3, max_tokens=2048
#     )
#     return resp.choices[0].message.content.strip()

# polishing_service.py

# app/services/polishing_service.py

from openai import OpenAI
import os

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")


def polish_text(text: str, language: str, prompt_template: str = None) -> str:
    """
    Полировка всего текста транскрипции.
    """
    if not _openai:
        return text
    if not prompt_template:
        prompt_template = (
            "Мына транскрипцияны тексерiп, қателерді түзетіңіз және форматтаңыз: {text}"
            if language == "kk"
            else "Check and correct this transcription, fix grammar and formatting: {text}"
        )
    prompt = prompt_template.format(text=text)
    resp = _openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": f"You are a helpful assistant improving {language} transcriptions."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2048
    )
    return resp.choices[0].message.content.strip()


def polish_segments(segments: list[dict], language: str) -> list[dict]:
    polished = []
    for idx, seg in enumerate(segments):
        # собираем контекст, но не включаем его в ответ
        prev = segments[idx - 1]['text'] if idx > 0 else None
        curr = seg['text']
        nxt  = segments[idx + 1]['text'] if idx < len(segments) - 1 else None

        # формируем только для отправки
        context_parts = []
        if prev: context_parts.append(f"Пред. сегмент: «{prev}»")
        context_parts.append(f"Тек. сегмент: «{curr}»")
        if nxt:  context_parts.append(f"След. сегмент: «{nxt}»")
        context = "\n".join(context_parts)

        prompt_template = (
            "У вас есть контекст диалога:\n"
            "{context}\n\n"
            "Отполируйте **только** текст текущего сегмента (то, что помечено «Тек. сегмент») "
            "и верните **исключительно** этот текст без каких-либо меток или дополнительного контекста."
        )

        # передаём context в текст промпта
        raw = prompt_template.format(context=context)
        polished_text = polish_text(
            text=raw,
            language=language,
            prompt_template="{text}"
        )

        polished.append({**seg, "polished_text": polished_text.strip()})
    return polished

