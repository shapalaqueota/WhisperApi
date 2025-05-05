import os
from speechbrain.inference.interfaces import foreign_class

# 1. Инициализация классификатора
# Не забыть скачать «custom_interface.py» из репозитория модели
classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    run_opts={"device": os.getenv("DEVICE_NAME", "cpu")}
)

def detect_emotion(audio_path: str) -> str:
    """
    Классифицирует файл audio_path и возвращает строку-метку эмоции.
    Если модель вернула список, берём первый элемент.
    """
    out_prob, score, idx, label = classifier.classify_file(audio_path)
    # если label — список или кортеж, разворачиваем
    if isinstance(label, (list, tuple)) and label:
        label = label[0]
    return str(label)
