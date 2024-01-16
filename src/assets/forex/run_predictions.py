import schedule


from AUDUSD.prediction_oop import run_predictions as USDAUD_P



def job():
    USDAUD_P()

schedule.every(10).minutes.do(job)

while True:
    schedule.run_pending()