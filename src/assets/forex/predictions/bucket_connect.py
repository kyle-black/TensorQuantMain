import boto3
from botocore.client import Config
import os

# Define your S3 URL and credentials
s3_url = 'https://tensorquantmodelstwo.nyc3.digitaloceanspaces.com'
access_key = 'DO00R8Y2RD2BNY3J2B44'
secret_key = 'VYzjlGfUV0VTta9TLh+Mp9kHc3c+VB3ISRFsWuL+69g'

# Create a session using your credentials
session = boto3.session.Session()

# Create an S3 client using your session
s3 = session.client('s3',
                    endpoint_url=s3_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    config=Config(signature_version='s3v4'))



# Define your local directory and S3 bucket
local_directory = 'models'
bucket = 'completedmodels'

# Walk through all files in the local directory
for root, dirs, files in os.walk(local_directory):
    for file in files:
        # Construct the full local path
        local_path = os.path.join(root, file)

        # Construct the full S3 path
        relative_path = os.path.relpath(local_path, local_directory)
        s3_path = os.path.join(bucket, relative_path)

        # Upload the file
        with open(local_path, 'rb') as data:
            s3.upload_fileobj(data, bucket, s3_path)