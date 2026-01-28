import os, sys
sys.path.insert(0,os.getcwd() )

import torch
import numpy as np
import json


def filter_foreigner_tokens(generated_ids, tokenizer, threshold=29920):
    """
    Filters out tokens with IDs greater than threshold, and replaces them with the pad token.
    Default threshold is where Cyrillic vocabulary starts.
    
    Args:
        generated_ids (torch.Tensor): The tensor containing generated token IDs.
        tokenizer: The tokenizer to use for fetching the pad token ID.
        threshold (int, optional): The threshold above which token IDs are considered spurious (default is 29920).
        
    Returns:
        torch.Tensor: The filtered tensor with spurious tokens replaced by pad tokens.
    """
    
    # Get the pad token ID from the tokenizer
    pad_token_id = tokenizer.pad_token_id
    
    # Create a copy of the generated_ids to modify
    filtered_ids = generated_ids.clone()
    
    # Replace spurious tokens (those greater than the threshold) with pad token ID
    filtered_ids[filtered_ids > threshold] = pad_token_id
    """
    # debug
    if (filtered_ids != generated_ids).any():
        print("\nSpurious characters were found. Replacing them with pad tokens.")
	"""
    return filtered_ids

def gradient_checkpoint_setup(args, model):
	# Enable gradient checkpointing to save memory
	if args.gradient_checkpointing:
		try:
			model.gradient_checkpointing_enable()
			model.config.use_cache = False  # Disable use_cache when gradient checkpointing is enabled
			model.is_gradient_checkpointing = True
		except Exception as e:
			# Catch any other exceptions and print the error message
			print(f"An unexpected error occurred: {e}. Continuing training.")
	else: 
		try:
			model.config.use_cache = True
			model.is_gradient_checkpointing = False
		except Exception as e:
			# Catch any other exceptions and print the error message
			print(f"An unexpected error occurred: {e}. Continuing training.")


# Function to batch the data
def batching(data_dict, batch_size):
    # Assuming all keys except 'return_loss' have values that are lists or tensors
    num_batches = len(data_dict['query']) // batch_size  # Use 'query' to determine the number of batches
    for i in range(num_batches):
        # Get a slice for each key except 'return_loss'
        batch = {
            key: values[i * batch_size: (i + 1) * batch_size] if isinstance(values, list) else values
            for key, values in data_dict.items()
            if key != 'return_loss'  # Exclude 'return_loss' from slicing
        }

        # Add the 'return_loss' value directly (no slicing)
        batch['return_loss'] = data_dict['return_loss']
        
        yield batch


def safe_update(stats, loss, kl, log_path, loss_values, kl_divergences):
	
	# Recursive function to convert ndarray to list
	def convert_ndarrays(obj):
		if isinstance(obj, dict):
			return {k: convert_ndarrays(v) for k, v in obj.items()}
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return obj
	
	if any(x is None for x in (stats, loss, kl)):
		return loss_values, kl_divergences, False
	else:
		data_serializable = convert_ndarrays(stats) # Convert the dictionary
		with open(log_path, 'a') as f:
			f.write(json.dumps(data_serializable) + "\n")	
		loss_values.append(loss)
		print("value head losses: ", loss_values)
		kl_divergences.append(kl)
		print("kl divergences: ", kl_divergences )
	
	return loss_values, kl_divergences, True

def subdivide_list(input_list, batch_size):
		"""Splits input_list into smaller batches of the specified batch_size."""
		return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


def create_prompt(instruction, fewshots=False):
	"""
	Helper function to create the prompt string based on the presence of input.

	Args:
	instruction: instruction from agents' prompt

	Returns:
	str: The formatted prompt string.
	"""
	facts= instruction.split("(END OF EXAMPLES)\n\n")[-1].split("\n\nPrice Movement:")[0].split("Facts:")[-1]  # facts only
	if fewshots:
		instruction= instruction.split("(END OF EXAMPLES)\n\n")[0]+ "(END OF EXAMPLES)"
	else: # instruction only
		instruction= instruction.split("Here are some examples:")[0] 
	incipit="Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
	return f"""{incipit}

### Instruction:
{instruction}

### Facts:
\n{facts}

### Response:
\n\nPrice Movement:
"""

def count_tokens(prompt, tokenizer):
	
	tokenized= tokenizer(prompt,
			truncation=False,  
			padding= False, 
			max_length= 2048, # to avoid default 
			return_tensors="pt",
			)
	num_tokens= tokenized["input_ids"].shape[1]
	return int(num_tokens)

	
