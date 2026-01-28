import os

def remove_checkpoint(parent_folder: str):
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

	# --- 2. Remove all files and subfolders to the parent folder ---
	for item in os.listdir(checkpoint_dir):
		file_path = os.path.join(checkpoint_dir, item)
		file_name=os.path.basename(file_path)
		print(f"Removing {file_name}")
		os.remove(file_path)

	# --- 3. Remove the empty checkpoint folder ---
	os.rmdir(checkpoint_dir)
	print(f"Removed folder: {checkpoint_dir}")
	return True
