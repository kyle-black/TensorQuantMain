'''
import os
import sys

current_directory = os.getcwd()
print(current_directory)

#forex_pairs = ['AUDUSD','USDJPY', 'NZDUSD', 'USDCAD', 'USDCHF']
#forex_pairs = ['AUDUSD','USDJPY', 'NZDUSD', 'USDCAD', 'USDCHF']
#forex_pairs = ['AUDUSD','EURUSD', 'USDCAD','GBPUSD']
forex_pairs = ['AUDUSD', 'USDCAD']
#### Tested: 'AUDUSD','EURUSD', 'GBPUSD'
#### not working: 'EURUSD' 'GBPUSD'

original_directory = os.getcwd()
# Add each forex pair's directory to the Python path
for pair in forex_pairs:
    os.chdir(os.path.join(current_directory, pair))
    print('first part:',os.getcwd())
    
    
    module = __import__(pair + '.prediction_oop', fromlist=[pair])
   # module.runner(pair)
    module.runner(pair)
    os.chdir(original_directory)
    print('second part:',os.getcwd())
# Import the prediction_oop module from each forex pair's directory and run the runner function


for pair in forex_pairs:
    module = __import__(pair + '.prediction_oop', fromlist=[pair])
    module.runner(pair)
'''

import subprocess


# List of scripts to run
scripts = [
   #"AUDUSD/prediction_oop.py",
   "EURUSD/prediction_oop.py"
   
    
]

# Run each script in a separate subprocess
processes = [subprocess.Popen(["python", script]) for script in scripts]

# Wait for all processes to finish
for process in processes:
    process.wait()
