import smtplib
from email.message import EmailMessage
import mimetypes
import zipfile
import os

def run_email():
    # Email setup
    sender_email = "kpblack87@gmail.com"
    receiver_email = "kpblack87@gmail.com"
    password = "tgmq pkpa oquz zhwi"  # Consider using a more secure authentication method
    subject = "Test Data CSV"
    body = "Attached is the zipped file containing test_data.csv and updated_df2.csv."
    
    # Files to attach
    files = ["test_data.csv", "updated_df2.csv"]
    zip_filename = "data_files.zip"

    # Create a zip file
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in files:
            zipf.write(file)

    # Create the email message
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    # Attach the zip file
    with open(zip_filename, "rb") as f:
        file_data = f.read()
        file_name = f.name
        msg.add_attachment(file_data, maintype="application", subtype="zip", filename=file_name)

    # Send the email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:  # Use your SMTP server
        server.login(sender_email, password)
        server.send_message(msg)

    # Clean up the zip file
    os.remove(zip_filename)

# Run the function
run_email()


