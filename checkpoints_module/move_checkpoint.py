import os
import shutil

def move_checkpoint(parent_folder: str):
	"""
	Finds a folder named 'checkpoint-N' inside `parent_folder`,
	moves all its contents to the parent folder, and removes the 
	empty checkpoint directory.

	Args:
		parent_folder (str): Path to the directory containing checkpoint-N.
	"""
	# --- 1. Find the folder named "checkpoint-N" ---
	checkpoint_dir = None
	for name in os.listdir(parent_folder):
		full_path = os.path.join(parent_folder, name)
		if name.startswith("checkpoint-") and os.path.isdir(full_path):
			checkpoint_dir = full_path
			break

	if checkpoint_dir is None:
		print("No checkpoint-N folder found in the specified parent folder.")
		return False

	print(f"Found checkpoint folder: {checkpoint_dir}")

	# --- 2. Move all files and subfolders to the parent folder ---
	for item in os.listdir(checkpoint_dir):
		src = os.path.join(checkpoint_dir, item)
		dst = os.path.join(parent_folder, item)
		print(f"Moving {src} → {dst}")
		shutil.move(src, dst)

	# --- 3. Remove the empty checkpoint folder ---
	os.rmdir(checkpoint_dir)
	print(f"Removed folder: {checkpoint_dir}")
	return True
