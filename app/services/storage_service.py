# app/services/storage_service.py
import boto3
import uuid
import os
from fastapi import UploadFile
import logging
from dotenv import load_dotenv
from botocore.client import Config

load_dotenv()
logger = logging.getLogger(__name__)

class S3StorageService:
    def __init__(self):
        self.access_key = os.getenv('S3_ACCESS_KEY')
        self.secret_key = os.getenv('S3_SECRET_KEY')
        self.endpoint_url = os.getenv('S3_ENDPOINT_URL')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')

        self._validate_config()

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            endpoint_url=self.endpoint_url,
            region_name='us-east-1',
            config=Config(signature_version='s3v4'),
            # verify=False  # Disable SSL verification
        )

    def _validate_config(self):
        missing = []
        for key in ['S3_ACCESS_KEY', 'S3_SECRET_KEY', 'S3_ENDPOINT_URL', 'S3_BUCKET_NAME']:
            if not os.getenv(key):
                missing.append(key)

        if missing:
            error_msg = f"Missing required S3 configuration: {', '.join(missing)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def upload_file(self, file: UploadFile) -> dict:
        try:
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"

            await file.seek(0)
            file_content = await file.read()
            file_size = len(file_content)

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=unique_filename,
                Body=file_content,
                ContentType=file.content_type
            )

            # Construct file URL
            file_url = f"{self.endpoint_url}/{self.bucket_name}/{unique_filename}"

            return {
                "original_filename": file.filename,
                "s3_filename": unique_filename,
                "s3_url": file_url,
                "size": file_size
            }

        except Exception as e:
            logger.error(f"S3 upload error: {str(e)}")
            raise

storage_service = S3StorageService()