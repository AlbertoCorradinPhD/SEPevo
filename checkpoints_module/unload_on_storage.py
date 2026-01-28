import subprocess
import os

def unload_on_storage(run_folder="toy_run", dest_folder="/content/destination_folder"):
    """
    Copies ONLY new files from run_folder into dest_folder/run_folder_name
    without overwriting any existing files.
    """

    run_folder_name = os.path.basename(run_folder.rstrip("/"))
    
    bash_script_content = f"""#!/bin/bash

# Ensure destination folder structure exists
mkdir -p {dest_folder}/{run_folder_name}

# Copy ONLY new files from run_folder → dest_folder/run_folder
# -n = do not overwrite existing files
cp -rn {run_folder}/* {dest_folder}/{run_folder_name}/
"""

    # Write bash script
    bash_script_path = "/content/unload_on_storage.sh"
    with open(bash_script_path, "w") as f:
        f.write(bash_script_content)

    # Make executable
    subprocess.run(["chmod", "+x", bash_script_path])

    # Execute script
    subprocess.run([bash_script_path])

    print(f"Bash script {bash_script_path} executed successfully.")


