import sys
import os

current_directory = os.getcwd()

print(current_directory)
'''
forex_pairs = ['AUDUSD', 'GBPUSD', 'USDCHF', 'USDHKD', 'EURUSD', 'USDCAD', 'USDCNH', 'USDJPY']

# Add each forex pair's directory to the Python path
for pair in forex_pairs:
    sys.path.append(f'/path/to/{pair}')

# Import the prediction_oop module from each forex pair's directory and run the runner function
for pair in forex_pairs:
    module = __import__(pair + '.prediction_oop', fromlist=[pair])
    module.runner(pair)
'''