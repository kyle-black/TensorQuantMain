import schedule


from prediction_oop import run_predictions



#def job(symbol):
#run_predictions(symbol)



#symbol = 'EURUSD'
#schedule.every(1).minutes.do(job(symbol))

#while True:
#    schedule.run_pending()


if __name__ in "__main__":
    symbol= 'EURUSD'
    run_predictions(symbol)
    #job(symbol)