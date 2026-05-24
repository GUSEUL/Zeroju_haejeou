import os
import time
import sys

# Files to monitor
watch_files = {
    'mdc_mdp_env.py': 'CODE',
    'build_mdp_model.py': 'CODE',
    'build_mdp_model_analytical.py': 'CODE',
    'results/mdp_formulation_ko.md': 'DOC'
}

# Resolve absolute paths
workspace_dir = os.path.dirname(os.path.abspath(__file__))
resolved_watch = {}
for rel_path, file_type in watch_files.items():
    abs_path = os.path.join(workspace_dir, rel_path.replace('/', os.sep))
    resolved_watch[abs_path] = {
        'rel_path': rel_path,
        'type': file_type,
        'last_mtime': os.path.getmtime(abs_path) if os.path.exists(abs_path) else 0.0
    }

print("Watcher started. Monitoring changes...", flush=True)

try:
    while True:
        time.sleep(2)
        for abs_path, info in resolved_watch.items():
            if not os.path.exists(abs_path):
                # If file doesn't exist yet but might be created later
                continue
            
            current_mtime = os.path.getmtime(abs_path)
            if current_mtime > info['last_mtime']:
                # Avoid triggering on the very first loop if last_mtime was 0.0 but it existed
                old_mtime = info['last_mtime']
                info['last_mtime'] = current_mtime
                if old_mtime > 0.0:
                    print(f"[CHANGE_DETECTED] Type: {info['type']}, File: {info['rel_path']}", flush=True)
except KeyboardInterrupt:
    print("Watcher stopped.", flush=True)
