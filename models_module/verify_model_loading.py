import os, sys
sys.path.insert(0, os.getcwd())

from utils.prompts import DEFAULT_INSTANCE_CAUSAL_MODEL, PREDICT_INSTRUCTION, DEFAULT_FACTS_SEQCLASS_MODEL, DEFAULT_RESPONSE_J, DEFAULT_RESPONSE_K
from toolbox.miscellaneous import filter_foreigner_tokens, create_prompt
from models_module.transform_reward_to_sentiment import transform_reward_to_sentiment

import torch
from transformers import AutoTokenizer, pipeline
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    TaskType,
)

import copy
import random
from datetime import datetime

def verify_causal_model(model, tokenizer, args, generation_kwargs, VH=False):
	"""
	Function to test whether the PEFT (LoRA) model has been loaded correctly.
	It checks if the model is functional and if LoRA layers are applied correctly.

	Args:
		model: The loaded PEFT model.
		tokenizer: Tokenizer associated with the model.
		test_input: A sample input text for testing.

	Returns:
		A boolean indicating whether the model is correctly loaded and functional.
	"""
	print("\nTesting the following model:")
	# For debug
	# print(model)

	try:
		# Step 1: Check model architecture (specifically looking for LoRA layers)
		lora_layers = [name for name, _ in model.named_parameters() if 'lora' in name.lower()]
		
		if len(lora_layers) == 0:
			print("Warning: No LoRA layers found in the model. Ensure LoRA is applied correctly.")
		else:
			print(f"LoRA layers found: {lora_layers}")
			"""
			# Check values assigned to parameters
			for name, param in model.named_parameters():
				if "lora" in name.lower():   # filter only LoRA layers
					print(f"\n=== {name} ===")
					print("Tensor:", param)
					print("Values:", param.data)
			"""
		device = next(model.parameters()).device

		# Step 2: Set the seed for random number generation globally
		current_time = datetime.now()
		seed = int(current_time.timestamp())  # Use current timestamp for a unique seed
		random.seed(seed)  # Set random seed
		torch.manual_seed(seed)  # Set PyTorch seed

		# Step 3: Tokenize input and perform a forward pass
		tokenized = tokenizer(
			create_prompt(DEFAULT_INSTANCE_CAUSAL_MODEL), 
			truncation=False,  # Don't truncate, keep the full sequence
			padding=args.padding,  
			pad_to_multiple_of=args.pad_to_multiple_of, 
			padding_side='left',  # For inference
			return_tensors='pt',
			max_length= args.cutoff_len if args.max_len_rl is None else min(args.cutoff_len, args.max_len_rl),
		)

		# Step 4: Perform inference
		print("Performing inference:")
		with torch.no_grad():
			generated_ids = model.generate(
				input_ids=tokenized['input_ids'].to(device),
				attention_mask=tokenized['attention_mask'].to(device),
				**generation_kwargs,
			)

		filtered_ids_tensor = filter_foreigner_tokens(generated_ids.squeeze(), tokenizer)
		text = tokenizer.decode(filtered_ids_tensor, skip_special_tokens=True)

		# Step 5: Decode the output
		print(f"Generated Output: {text}")
		
		if VH:
			# Step 6: Perform inference and inspect values on the head:
			print("\nVerify output from the value head:")
			with torch.no_grad():
				logits, _, values = model(
					input_ids=tokenized['input_ids'].to(device),
					attention_mask=tokenized['attention_mask'].to(device),
				)
			print("Values:")
			print(values)

		# If we reach this point, the model was able to load and generate output
		return True

	except Exception as e:
		print(f"Error running the model: {e}")
		return False



def verify_seqClass_model(model, tokenizer, args, test_input="this movie was really bad!!" ):
	"""
	Verifies the reward model, processes the logits, and transforms them into sentiment output.
	
	Args:
		model: The PEFT model.
		tokenizer: Tokenizer associated with the model.
		test_input: The input string to test.
		
	Returns:
		dict: The transformed sentiment label and score.
	"""
	print("\nTesting the following model:")
	# for debug
	#print(model)
	
	try:
		# Step 1: Check model architecture (specifically looking for LoRA layers)
		lora_layers = [name for name, _ in model.named_parameters() if 'lora' in name.lower()]
		
		if len(lora_layers) == 0:
			print("Warning: No LoRA layers found in the model. Ensure LoRA is applied correctly.")
		else:
			print(f"LoRA layers found: {lora_layers}")

		device = next(model.parameters()).device
		
		tokenized = tokenizer(test_input,
			truncation=False,  # Don't truncate, keep the full sequence
			padding=args.padding,  
			pad_to_multiple_of=args.pad_to_multiple_of,  
			padding_side='left', # for inference
			max_length= args.cutoff_len if args.max_len_rm is None else min(args.cutoff_len, args.max_len_rm),
			return_tensors='pt',
			)
		
		# Step 2: Set the seed for random number generation globally
		current_time = datetime.now()
		seed = int(current_time.timestamp())  # Use current timestamp for a unique seed
		random.seed(seed)  # Set random seed
		torch.manual_seed(seed)  # Set PyTorch seed
	
		# Step 3: Perform inference
		print("Performing inference:")
		with torch.no_grad():
			logits = model(input_ids=tokenized['input_ids'].to(device), attention_mask=tokenized['attention_mask'].to(device)).logits.item()
		print(f"Logit output: {logits}")
		
		# Step 4: Transform logit to sentiment label
		sentiment = transform_reward_to_sentiment(logits)
		print(f"Transformed Sentiment Output: {sentiment}")
		
		return sentiment

	except Exception as e:
		print(f"Error loading or running the model: {e}")
		return None

def verify_reward_model(model, tokenizer, args):
	# test trained reward model
	instruction= PREDICT_INSTRUCTION.split("Here are some examples")[0]
	test_input1="Instruction: " + instruction + "\n\nFacts:\n"+DEFAULT_FACTS_SEQCLASS_MODEL+ "\n\nPrice Movement: " + DEFAULT_RESPONSE_J
	test_input2="Instruction: " + instruction + "\n\nFacts:\n"+DEFAULT_FACTS_SEQCLASS_MODEL+ "\n\nPrice Movement: " + DEFAULT_RESPONSE_K
	print("Test the model for rewards with a use case:")
	print("Test 1:\n")
	_= verify_seqClass_model(model, tokenizer, args, test_input=test_input1)
	print("Test 2:\n")
	_= verify_seqClass_model(model, tokenizer, args, test_input=test_input2)
