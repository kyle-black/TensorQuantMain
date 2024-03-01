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
#bucket = 'testcompletedmodels'

bucket = 'tensorquantmodelstwo'




def download_files_from_space(space_folder='', bucket = 'tensorquantmodelstwo'):
    
    # List all objects in the space folder
    response = s3.list_objects_v2(Bucket=bucket, Prefix=space_folder)

    # Check if there are any objects
    if 'Contents' in response:
        objects = response['Contents']

        # Download each object
        for obj in objects:
            key = obj['Key']
            if key.endswith('/'):
                # It's a folder, create local folder and recursively download its content
                folder_name = key.rstrip('/')
                next_space_folder = os.path.join(space_folder, folder_name)
                download_files_from_space(next_space_folder)
            else:
                # It's a file, download it
               # local_file_path = os.path.join(LOCAL_DIRECTORY, key)
               # os.makedirs(os.path.dirname(local_file_path), exist_ok=True)  # Create parent directory if not exists
                print(f"Downloading {key}...")
               # s3.download_file(SPACE_NAME, key, local_file_path)
             #   print(f"Downloaded {key} to {local_file_path}")
    else:
        print(f"No objects found in folder: {space_folder}")

if __name__ == "__main__":
    download_files_from_space()




# Walk through all files in the local directory

'''
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
'''

#response = s3.list_objects_v2(Bucket=bucket)

#for item in response['Contents']:
#    print(item['Key'])

#response = s3.list_buckets()

#print(response)
#for bucket in response['Buckets']:
#    print(bucket['Name'])