import boto3
from botocore.client import Config

# Define your S3 URL and credentials
s3_url = 'https://tensorquantmodels.nyc3.digitaloceanspaces.com'
access_key = 'your_access_key'
secret_key = 'your_secret_key'

# Create a session using your credentials
session = boto3.session.Session()

# Create an S3 client using your session
s3 = session.client('s3',
                    endpoint_url=s3_url,
                    #aws_access_key_id=access_key,
                   # aws_secret_access_key=secret_key,
                    config=Config(signature_version='s3v4'))

# Upload the model file to your S3 bucket
with open("../models/AUDUSD_models/random_forest_model_up_AUDUSD_60.pkl", "rb") as data:
    s3.upload_fileobj(data, 'mybucket', 'random_forest_model_up_AUDUSD_60.pkl')