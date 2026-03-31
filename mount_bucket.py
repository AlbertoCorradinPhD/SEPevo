import os
import subprocess

def mount_bucket(bucket_name, mount_path):
    # 1. Create the local directory if it doesn't exist
    if not os.path.exists(mount_path):
        os.makedirs(mount_path)
        print(f"Created directory: {mount_path}")

    # 2. Construct the gcsfuse command
    # --implicit-dirs ensures folders show up correctly
    command = [
        "gcsfuse",
        "--implicit-dirs",
        bucket_name,
        mount_path
    ]

    try:
        # 3. Execute the mount
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Success! Bucket '{bucket_name}' is now mounted at '{mount_path}'")
        print(f"You can now access your scripts in: {os.path.abspath(mount_path)}")
    except subprocess.CalledProcessError as e:
        print(f"Error mounting bucket: {e.stderr}")

# --- CONFIGURATION ---
MY_BUCKET = "sep_evo_trial"
LOCAL_FOLDER = "SEPevo" # This will appear in your current directory

mount_bucket(MY_BUCKET, LOCAL_FOLDER)
