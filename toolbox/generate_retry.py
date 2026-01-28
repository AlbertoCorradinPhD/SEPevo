import os, sys
sys.path.insert(0,os.getcwd() )

from toolbox.miscellaneous import filter_foreigner_tokens

import torch
import random
import time
from datetime import datetime


def generate_retry(model, tokenizer, batch, generation_kwargs={}, max_retries=5):
	
	device = next(model.parameters()).device
	response_tensors = []
	L=[]

	for i, (input_ids, attention_mask) in enumerate( zip(batch["input_ids"], batch["attention_mask"]) ):
		
		print("Query number:", str(i + 1))
		success = False

		for attempt in range(max_retries):
			try:
				# New seed for each retry
				seed = int(time.time()) + i + attempt
				random.seed(seed)
				torch.manual_seed(seed)

				with torch.no_grad():
					generated_ids = model.generate(
						input_ids=input_ids.to(device),
						attention_mask=attention_mask.to(device),
						**generation_kwargs,
					)

				num_tokens= input_ids.shape[1]
				response= filter_foreigner_tokens( generated_ids[:, num_tokens:].squeeze(), tokenizer)
				L.append(len(response))
				# debug
				#print("Debug. Length of the responses: ", L)
				success = True
				response_tensors.append(response)
				break  # Exit retry loop on success

			except Exception as e: # Retry condition
				print( f"Retry {attempt + 1}/{max_retries} failed for batch item {i}: {e}"	)
				continue

		if not success:
			print(f"Generation failed after {max_retries} retries for batch item {i}")
			return None  # or append a placeholder instead

	return response_tensors
