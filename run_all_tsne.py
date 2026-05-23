import subprocess
import concurrent.futures
import time

lambdas = [0.1, 0.5, 1.5, 3.0]
reward_types = ["standard", "sparse", "cliff"]
episodes = 30000

def run_tsne(lambda_val, reward_type):
    start = time.time()
    print(f"Starting t-SNE: Lambda={lambda_val}, Reward={reward_type}")
    cmd = [
        "python", 
        "visualization/plot_tsne_embedding.py", 
        f"--lambda_val={lambda_val}", 
        f"--reward_type={reward_type}",
        f"--episodes={episodes}"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running t-SNE for L={lambda_val}, R={reward_type}:\n{res.stderr}")
        return False
    elapsed = time.time() - start
    print(f"Finished t-SNE: Lambda={lambda_val}, Reward={reward_type} in {elapsed:.2f} seconds")
    return True

if __name__ == "__main__":
    t_start = time.time()
    scenarios = [(l, r) for l in lambdas for r in reward_types]
    
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_tsne, l, r) for l, r in scenarios]
        for f in concurrent.futures.as_completed(futures):
            f.result()
            
    print(f"All t-SNE plots generated in {time.time() - t_start:.2f} seconds.")
