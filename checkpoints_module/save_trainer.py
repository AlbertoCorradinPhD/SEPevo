import os
import torch
from transformers import Trainer
from peft import PeftModel
import json

def save_trainer(trainer, output_dir: str):
	"""
	Save the model and trainer state. If the model is a PEFT model, save it as such.
	If the model is not a PEFT model, save the entire model as usual.
	
	Args:
		trainer (Trainer): The Hugging Face Trainer object.
		output_dir (str): Directory where the model and trainer state will be saved.
		is_peft_model (bool, optional): Flag to indicate whether the model is a PEFT model (default: False).
	
	Returns:
		bool: Whether the save operation was successful (True/False).
	"""
	
	# Ensure the output directory exists
	os.makedirs(output_dir, exist_ok=True)
	done = False

	# Step 1: Save Trainer State
	try:
		if trainer.state is not None:
			trainer.save_state()  # Saves the trainer state to the `trainer.args.output_dir`
			print("Trainer state saved.")
		else:
			print("Trainer state is not available.")
	except Exception as e:
		print(f"Error while saving the trainer state: {str(e)}")

	# Step 2: Save Model
	try:
		if trainer.model is not None:
			trainer.model.save_pretrained(output_dir)
			print(f"PEFT (LoRA) adapters saved to {output_dir}.")
			done= True						
			model_config = trainer.model.config.to_dict()  # Assuming the model has a config with to_dict()
			file_path= os.path.join(output_dir, "config.json")
			with open(file_path, "w") as f:
				json.dump(model_config, f, indent=4)
			print(f"Model configuration saved to {file_path}.")
			done= True	
		else:
			print("Model is not available for saving.")
	except Exception as e:
		print(f"Error while saving the model: {str(e)}")

	# Step 3: Save Tokenizer (if available)
	try:
		if trainer.tokenizer is not None:
			trainer.tokenizer.save_pretrained(output_dir)
			print(f"Tokenizer saved to {output_dir}.")
	except Exception as e:
		print(f"Error while saving the tokenizer: {str(e)}")

	# Step 4: Save Optimizer
	try:
		if trainer.optimizer is not None:
			torch.save(trainer.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
			print(f"Optimizer state saved to {output_dir}.")
	except Exception as e:
		print(f"Error while saving optimizer: {str(e)}")

	# Step 5: Save Scheduler State
	try:
		if trainer.lr_scheduler is not None:
			torch.save(trainer.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
			print(f"Scheduler state saved to {output_dir}.")
	except Exception as e:
		print(f"Error while saving scheduler: {str(e)}")

	return done
