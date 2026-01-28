import os, sys
sys.path.insert(0,os.getcwd() )

from toolbox.miscellaneous import filter_foreigner_tokens

import torch
import random
import time
from datetime import datetime

def generate_once(model, tokenizer, batch, generation_kwargs={}):
	device = next(model.parameters()).device

	# Loop through each input in the batch and generate with a new seed per item
	generated_ids = []
	num_tokens = []
	
	for i, (input_ids, attention_mask) in enumerate(zip(batch["input_ids"], batch["attention_mask"])):
		#print(input_ids)
		print("Query number: ", str(i+1))		
		try:
			# Get current time and create a new seed for each generation
			current_time = datetime.now()
			seed = int(current_time.timestamp()) + i  # Ensure different seeds by adding i (index of batch)
			# Set the seed for random number generation globally
			random.seed(seed)
			torch.manual_seed(seed)

			# Generate the response for this input
			with torch.no_grad():
				generated_id = model.generate(
					input_ids=input_ids.to(device),
					attention_mask=attention_mask.to(device),
					**generation_kwargs,
					)
				
			generated_ids.append(generated_id)
			num_tokens.append(input_ids.shape[1])
				# debug
				#print(num_tokens)

		except Exception as e:
			print(f"Error during generation for batch item {i}: {e}")
			return None  # Return None on failure for this particular batch item
						
	# Process the generated ids to extract the responses
	responses_only = [t[:, n:] for t,n in zip(generated_ids,num_tokens)]
	response_tensors = [filter_foreigner_tokens(r.squeeze(), tokenizer) for r in responses_only]
	# Return the responses after processing
	return response_tensors
