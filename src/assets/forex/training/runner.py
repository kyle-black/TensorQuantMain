import os
import sys

current_directory = os.getcwd()
print(current_directory)

#forex_pairs = ['AUDUSD','USDJPY', 'NZDUSD', 'USDCAD', 'USDCHF']
#forex_pairs = ['AUDUSD','USDJPY', 'NZDUSD', 'USDCAD', 'USDCHF']
forex_pairs = ['USDCHF']

#### Tested: 'AUDUSD','EURUSD', 'GBPUSD'
#### not working: 'EURUSD' 'GBPUSD'


# Add each forex pair's directory to the Python path
for pair in forex_pairs:
    sys.path.append(os.path.join(current_directory, pair))

# Import the prediction_oop module from each forex pair's directory and run the runner function
for pair in forex_pairs:
    module = __import__(pair + '.prediction_oop', fromlist=[pair])
    module.runner(pair)