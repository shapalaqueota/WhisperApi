from pyannote.audio import Pipeline as PyannotePipeline
import os

# инициализация
diarizer = PyannotePipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HF_TOKEN"),
)
MIN_SEGMENT_LEN = 0.5
MERGE_GAP = 0.2

def diarize_file(audio_path: str):
    ann = diarizer({"audio": audio_path})
    raw = [{"start": round(t.start,2), "end": round(t.end,2), "speaker": spk}
           for t,_,spk in ann.itertracks(yield_label=True)]
    # фильтр и слияние близких сегментов
    filt = [s for s in raw if s['end']-s['start']>=MIN_SEGMENT_LEN]
    merged = []
    for seg in sorted(filt, key=lambda x: x['start']):
        if merged and seg['speaker']==merged[-1]['speaker'] \
           and seg['start']-merged[-1]['end']<=MERGE_GAP:
            merged[-1]['end'] = seg['end']
        else:
            merged.append(seg.copy())
    return merged
