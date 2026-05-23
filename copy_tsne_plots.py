import os
import shutil
import glob

src_pattern = "results/L_*_E_*/*/policy_tsne_*.png"
dest_dir = "visualization/tsne_plots"
os.makedirs(dest_dir, exist_ok=True)

files = glob.glob(src_pattern)
print(f"Found {len(files)} t-SNE plot files to consolidate.")

for f in files:
    # Example path: results\L_0.5_E_30000\cliff\policy_tsne_expected_sarsa.png
    parts = f.replace("\\", "/").split("/")
    folder_l_e = parts[1] # L_0.5_E_30000
    reward = parts[2] # cliff
    filename = parts[3] # policy_tsne_expected_sarsa.png
    
    # Extract lambda from L_0.5_E_30000 -> 0.5
    l_val = folder_l_e.split("_")[1]
    
    # Extract agent name from policy_tsne_expected_sarsa.png -> expected_sarsa
    agent = filename.replace("policy_tsne_", "").replace(".png", "")
    
    new_filename = f"tsne_L{l_val}_{reward}_{agent}.png"
    dest_path = os.path.join(dest_dir, new_filename)
    
    shutil.copy2(f, dest_path)
    print(f"Copied {f} -> {dest_path}")

print("Consolidation complete.")
