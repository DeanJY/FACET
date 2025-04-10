import os
import boto3
from pytorch_lightning.callbacks import ModelCheckpoint
from botocore.exceptions import ClientError

class S3Checkpoint(ModelCheckpoint):
    def __init__(self, s3_bucket, s3_prefix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3')

    def _upload_to_s3(self, file_path):
        try:
            s3_path = f"{self.s3_prefix}/{os.path.basename(file_path)}"
            self.s3_client.upload_file(file_path, self.s3_bucket, s3_path)
            print(f"Uploaded {file_path} to s3://{self.s3_bucket}/{s3_path}")
        except ClientError as e:
            print(f"Failed to upload {file_path} to S3: {e}")

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        # 防止重复上传相同的文件
        if self.last_model_path and self.last_model_path != self.best_model_path:
            self._upload_to_s3(self.last_model_path)

        if self.best_model_path:
            self._upload_to_s3(self.best_model_path)