import boto3

# Create an S3 client



s3_url = 'https://tensorquantmodels.nyc3.digitaloceanspaces.com'

s3 = boto3.client(s3_url)

# Upload the model file to your S3 bucket





with open("/models/AUDUSD_models/random_forest_model_up_AUDUSD_60.pkl", "rb") as data:
    s3.upload_fileobj(data, 'mybucket', 'random_forest_model_up_AUDUSD_60.pkl')