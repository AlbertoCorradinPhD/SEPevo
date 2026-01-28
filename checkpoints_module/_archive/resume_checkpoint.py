import os
import json
import warnings
from peft import set_peft_model_state_dict
import torch

def resume_checkpoint(dir_path,now_max_steps):
	try:
		# Check the available weights and load them
		checkpoint_name = os.path.join(
			dir_path, "pytorch_model.bin"
			)  # Full checkpoint
		if not os.path.exists(checkpoint_name):
			pytorch_bin_path = checkpoint_name
			checkpoint_name = os.path.join(
				dir_path, "adapter_model.bin"
				)  # only LoRA model - LoRA config above has to fit
		if os.path.exists(checkpoint_name):
			os.rename(checkpoint_name, pytorch_bin_path)
			warnings.warn(
				"The file name of the lora checkpoint'adapter_model.bin' is replaced with 'pytorch_model.bin'")
		else:
			dir_path = ( None )  # So the trainer won't try loading its state  
	except:
		print("Can't format path to checkpoint")
		return 

	# The two files above have a different name depending on how they were saved, but are actually the same.
	if os.path.exists(checkpoint_name):
		print(f"Loading previous checkpoint: {checkpoint_name}")
		try:
			adapters_weights = torch.load(checkpoint_name) 
			model = set_peft_model_state_dict(model, adapters_weights)
		except:
			print("Problems with 'set_peft_model_state_dict() function'")
			return None, None
			
		# Proceeds only if you could get the model checkpoint
		MAX_STEPS = now_max_steps
		try:
			train_args_path = os.path.join(
				dir_path, "trainer_state.json") # Corrected variable name
			if os.path.exists(train_args_path):
				base_train_args = json.load(open(train_args_path, 'r'))
				base_max_steps = base_train_args["max_steps"]
				# resume_scale = base_max_steps / now_max_steps # Unclear
				if base_max_steps > now_max_steps:
					warnings.warn("epoch {} replace to the base_max_steps {}".format(
						EPOCHS, base_max_steps))
					EPOCHS = None
					MAX_STEPS = base_max_steps  # This is where MAX_STEPS could become None if base_max_steps was None
					#here it should return model, base_train_args, MAX_STEPS, EPOCHS, base_train_args
					#checkpoint_info=
		except:
			print("Can't load the trainer state")
			return model, MAX_STEPS
	else:
		print(f"Checkpoint {checkpoint_name} not found")
		return None, None

		
