import subprocess
import os

def decompress_checkpoints(file_name, run_folder, dest_folder):
	
	#file_name = os.path.basename(file_path.rstrip("/"))

	bash_script_content = f"""#!/bin/bash
set -e  # stop on first error

# Install zstd
apt-get install -y zstd

# Ensure destination exists
mkdir -p {dest_folder}

SRC="{run_folder}/checkpoints/{file_name}.tar.zst"
DST="{dest_folder}/{file_name}.tar.zst"

# Verify archive exists
if [[ ! -f "$SRC" ]]; then
	echo "ERROR: Archive not found: $SRC"
	exit 1
fi

# Copy archive (overwrite allowed but explicit)
cp "$SRC" "$DST"

# Extract
cd {dest_folder}

# Safe extraction (no overwrite existing files)
tar --skip-old-files -I zstd -xf {file_name}.tar.zst

echo "Decompression completed successfully."
"""

	bash_script_path = "/content/decompress_checkpoint.sh"

	with open(bash_script_path, "w") as f:
		f.write(bash_script_content)

	subprocess.run(["chmod", "+x", bash_script_path])

	# Capture failures properly
	try:
		subprocess.run([bash_script_path], check=True)
	except subprocess.CalledProcessError as e:
		print(f"Decompression failed with code {e.returncode}")
		return False

	print(f"Bash script {bash_script_path} executed successfully.")
	return True

