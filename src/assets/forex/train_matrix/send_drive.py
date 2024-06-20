from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def upload_file_to_google_drive(file_path):
    # Authenticate and create the PyDrive client
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication
    drive = GoogleDrive(gauth)

    # Upload the file to Google Drive
    file = drive.CreateFile({'title': file_path})
    file.SetContentFile(file_path)
    file.Upload()

# Use the function
file_path = 'test_data.csv'
upload_file_to_google_drive(file_path)