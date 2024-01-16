import schedule


from predictions.prediction_oop import run_predictions



def job(symbol):
    run_predictions()



symbol = 'USDEUR'
schedule.every(10).minutes.do(job(symbol))

while True:
    schedule.run_pending()