import os

files = [
    '../MDP/build_mdp_model.py',
    '../MDP/train_all_mdp.py',
    '../MDP/visualize_mdp_results.py',
    '../MDP/mdc_mdp_env.py'
]

for f_path in files:
    if os.path.exists(f_path):
        with open(f_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace literal backslash-quote with just quote
        # This fixes the shell escaping artifacts
        fixed_content = content.replace('\\"', '"').replace("\\'", "'")
        
        with open(f_path, 'w', encoding='utf-8', newline='') as f:
            f.write(fixed_content)
        print(f"Fixed: {f_path}")
    else:
        print(f"Not found: {f_path}")
