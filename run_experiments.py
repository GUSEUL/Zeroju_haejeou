import subprocess
import concurrent.futures
import time
import sys

lambdas = [0.5, 1.5, 3.5]
reward_types = ["standard", "sparse", "cliff"]
episodes = 30000

def run_scenario(lambda_val, reward_type):
    start = time.time()
    print(f"Starting scenario: Lambda={lambda_val}, Reward={reward_type}", flush=True)
    
    # Step 1: Build model
    cmd1 = ["python", "build_mdp_model.py", f"--lambda_val={lambda_val}", f"--reward_type={reward_type}"]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    if res1.returncode != 0:
        print(f"Error building model for L={lambda_val}, R={reward_type}:\n{res1.stderr}", flush=True)
        return (lambda_val, reward_type, -1)
        
    # Step 2: Train all MDP agents
    cmd2 = ["python", "train_all_mdp.py", f"--lambda_val={lambda_val}", f"--episodes={episodes}", f"--reward_type={reward_type}"]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    if res2.returncode != 0:
        print(f"Error training agents for L={lambda_val}, R={reward_type}:\n{res2.stderr}", flush=True)
        return (lambda_val, reward_type, -1)
        
    # Step 3: Visualize
    cmd3 = ["python", "visualize_mdp_results.py", f"--lambda_val={lambda_val}", f"--episodes={episodes}", f"--reward_type={reward_type}"]
    res3 = subprocess.run(cmd3, capture_output=True, text=True)
    if res3.returncode != 0:
        print(f"Error visualizing for L={lambda_val}, R={reward_type}:\n{res3.stderr}", flush=True)
        return (lambda_val, reward_type, -1)
        
    elapsed = time.time() - start
    print(f"Finished scenario: Lambda={lambda_val}, Reward={reward_type} in {elapsed:.2f} seconds", flush=True)
    return (lambda_val, reward_type, elapsed)

if __name__ == "__main__":
    import os
    t_start = time.time()
    scenarios = [(l, r) for l in lambdas for r in reward_types]
    
    num_workers = min(9, os.cpu_count() or 3)
    print(f"Starting execution of {len(scenarios)} scenarios in parallel with {num_workers} workers...", flush=True)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_scenario, l, r) for l, r in scenarios]
        for f in concurrent.futures.as_completed(futures):
            l, r, elapsed = f.result()
            if elapsed >= 0:
                print(f"Success: Scenario L={l}, R={r} done in {elapsed:.2f}s", flush=True)
            else:
                print(f"FAILED: Scenario L={l}, R={r}", flush=True)
                sys.exit(1)
            
    print(f"All scenarios completed successfully in {time.time() - t_start:.2f} seconds.", flush=True)
