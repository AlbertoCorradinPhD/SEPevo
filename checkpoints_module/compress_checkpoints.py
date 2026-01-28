import subprocess
import os

def compress_checkpoints(file_path, run_folder, models_folder):
    """
    Compresses a checkpoint folder, copies the compressed archive into {run_folder}/checkpoints/
    """

    run_folder_name = os.path.basename(run_folder.rstrip("/"))
    file_name = os.path.basename(file_path.rstrip("/"))

    bash_script_content = f"""#!/bin/bash
set -e  # stop on first error

# Install zstd package
apt-get install -y zstd

# Change to saved_models
cd {models_folder}

# Compress checkpoint folder
tar --zstd -cf {file_name}.tar.zst {file_name}

# Ensure checkpoints folder exists inside run_folder
mkdir -p {run_folder}/checkpoints

# Copy compressed checkpoint
cp {file_name}.tar.zst {run_folder}/checkpoints/{file_name}.tar.zst

"""

    # Write bash script
    bash_script_path = "/content/compress_checkpoint.sh"
    with open(bash_script_path, "w") as f:
        f.write(bash_script_content)

    # Make executable
    subprocess.run(["chmod", "+x", bash_script_path])

    # Execute script
    subprocess.run([bash_script_path])

    print(f"Bash script {bash_script_path} executed successfully.")


