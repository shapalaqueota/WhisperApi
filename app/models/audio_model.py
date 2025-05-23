from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class AudioTranscription(Base):
    __tablename__ = "audio_transcriptions"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, nullable=False)
    s3_filename = Column(String, nullable=False, unique=True)
    s3_url = Column(String, nullable=False)
    file_size = Column(Integer)  # in bytes
    duration = Column(Float)
    language = Column(String)
    transcription = Column(Text)
    diarization_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    formatted_transcription = Column(Text, nullable=True)
    speakers = Column(JSON, nullable=True)
    overall_emotion         = Column(String, nullable=True)
    polished_text           = Column(Text, nullable=True)
