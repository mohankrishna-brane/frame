import numpy as np

file_path = 'data/vectors/NH3775.npy'  # <--- REPLACE THIS

try:
    data = np.load(file_path, allow_pickle=True)
    
    print(f"\n--- Analyzing {file_path} ---")
    print(f"Shape: {data.shape}")
    print(f"Type:  {data.dtype}")
    print("Data:")
    print(data)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"Error reading file: {e}")