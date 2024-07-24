import numpy as np
import os

# Check if NumPy is using OpenBLAS
print(np.__config__.show())

# Set the number of threads for OpenMP
os.environ["OMP_NUM_THREADS"] = "4"

# Set the number of threads for MKL
#os.environ["MKL_NUM_THREADS"] = "4"