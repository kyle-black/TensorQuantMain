

import subprocess


# List of scripts to run
scripts = [
  # "AUDUSD/prediction_oop.py","EURUSD/prediction_oop.py",
   "pull_ohlc.py"
   
    
]

# Run each script in a separate subprocess
processes = [subprocess.Popen(["python", script]) for script in scripts]

# Wait for all processes to finish
for process in processes:
    process.wait()
