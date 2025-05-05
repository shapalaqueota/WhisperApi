from openai import OpenAI
import os

# Инициализация OpenAI клиента
_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")


def polish_text(text: str, language: str, prompt_template: str = None) -> str:
    """
    Полировка полноценного текста транскрипции.
    Используется в ветке без диаризации.
    """
    if not _openai:
        return text
    # Стандартный шаблон, если не передали свой
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
    """
    Полировка каждого сегмента с учётом реального корпуса через RAG‑поиск.
    Возвращает список тех же сегментов с полированным текстом.
    """
    from .retrieval_service import get_relevant_examples

    polished = []
    for seg in segments:
        text = seg.get('text', '')
        # Получаем примеры из корпуса
        examples = get_relevant_examples(text, k=3)
        block = "\n".join(f"- {ex}" for ex in examples)

        # Формируем промпт
        prompt = (
            "Вот примеры реального употребления казахского языка:\n"
            f"{block}\n\n"
            "Отполируйте следующий текст сегмента, сохраняя смысл и стиль:\n"
            f"\"{text}\""
        )
        # Вызываем GPT для полировки сегмента
        resp = _openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant improving {language} transcriptions."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        polished_text = resp.choices[0].message.content.strip()
        polished.append({**seg, "polished_text": polished_text})

    return polished
