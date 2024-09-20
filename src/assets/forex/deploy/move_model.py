import boto3
from botocore.client import Config
import os


def upload_file_to_spaces(file_path, space_name, region, access_key, secret_key):
    # Create a session using your DigitalOcean Spaces credentials
    session = boto3.session.Session()
    
    # Create a client to interact with DigitalOcean Spaces
    client = session.client('s3',
                            region_name=region,
                            endpoint_url=f'https://{space_name}.{region}.digitaloceanspaces.com',
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key)

    # Extract file name from file path
    file_name = os.path.basename(file_path)

    # Upload the file
    with open(file_path, 'rb') as data:
        client.upload_fileobj(data, space_name, file_name)

    print(f"{file_name} uploaded successfully to {space_name} in {region}.")

# Upload the file

if __name__ == "__main__":
    # Parameters
    file_path = 'models/EURUSD/EURUSD_0916-61.h5'  # Path to your file
    space_name = 'saved_models'  # Name of your Space
    region = 'nyc3'  # Region of your Space, e.g., 'nyc3'
    access_key = os.environ['access_key']  
    secret_key = os.environ['secret_key']  # Your Spaces Secret Key
    
    # Call the function to upload the file
    upload_file_to_spaces(file_path, space_name, region, access_key, secret_key)

