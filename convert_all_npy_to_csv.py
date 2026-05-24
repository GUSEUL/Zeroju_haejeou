import os
import numpy as np

print("Converting existing NPY Q-tables to CSV format...")
count = 0
for root, dirs, files in os.walk("results"):
    for file in files:
        if file.endswith(".npy") and file.startswith("q_table_"):
            npy_path = os.path.join(root, file)
            csv_path = npy_path.replace(".npy", ".csv")
            q = np.load(npy_path)
            np.savetxt(csv_path, q, delimiter=",")
            count += 1
            print(f" [{count}] Converted: {npy_path} -> {csv_path}")

print(f"Successfully converted {count} Q-tables to CSV.")
