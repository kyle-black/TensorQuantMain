import smtplib
from email.message import EmailMessage
import mimetypes





def run_email(model_num):
# Email setup#########
    sender_email = "kpblack87@gmail.com"
    receiver_email = "kpblack87@gmail.com"
    password = "tgmq pkpa oquz zhwi"  # Consider using a more secure authentication method
    subject = "Test Data CSV"
    body = "Attached is the test_data.csv file."
    file_path = f"test_result_{model_num}.csv"
   # file_path = "test_result_EURUSD_0816-31.csv"
   # file_path ='test2.zip'
    # Create the email message
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

     # Attach the file
    ctype, encoding = mimetypes.guess_type(file_path)
    if ctype is None or encoding is not None:
        ctype = 'application/octet-stream'
    maintype, subtype = ctype.split('/', 1)

    # Attach the file
    with open(file_path, "rb") as f:
        file_data = f.read()
        file_name = f.name
    msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)

    # Send the email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:  # Use your SMTP server
        server.login(sender_email, password)
        server.send_message(msg)


if __name__ in "__main__":
    run_email(None)