import numpy as np

# 1. Generate the 512-dimensional vector
vector_512 = np.random.rand(512)

# 2. Save to a text file
# option='%.6f' limits the precision to 6 decimal places to keep the file size manageable
# delimiter=',' separates the values with commas
np.savetxt("test_vector.txt", vector_512, fmt="%.6f", newline=",")

print("Successfully saved 512-dim vector to 'test_vector.txt'")
