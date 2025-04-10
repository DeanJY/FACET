import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.pytorch import PyTorch


# Initializes SageMaker session which holds context data
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name

# The bucket containig our input data
bucket = "s3://eveye-dataset-full"
output_path = "s3://junyuan1213/eveye"
checkpoint_local_path = "/home/junyuan/logs"
checkpoint_s3_uri = "s3://junyuan1213"
channels = {"root": bucket}

# sagemaker.get_execution_role()
role = "arn:aws:iam::339713122800:role/service-role/AmazonSageMaker-ExecutionRole-20240430T215474"

# Creates a new PyTorch Estimator with params
estimator = PyTorch(
    # name of the runnable script containing __main__ function (entrypoint)
    entry_point="tools/train.py",
    source_dir="./",
    role=role,
    framework_version="2.2.0",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g5.8xlarge",
    hyperparameters={},
    output_path=output_path,
    checkpoint_s3_uri=checkpoint_s3_uri,
    checkpoint_local_path=checkpoint_local_path,
    sagemaker_session=sagemaker_session,
)

# Call fit method on estimator, wich trains our model, passing training
# and testing datasets as environment variables. Data is copied from S3
# before initializing the container
estimator.fit(channels)
