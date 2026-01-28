from functools import lru_cache
import torch

@lru_cache(maxsize=100) 
def calculate_rewards(reward_model, tokenizer, texts, padding, max_len, pad_to_multiple_of):
	
	def parse_text_and_tokenize(text, tokenizer, padding, max_len, pad_to_multiple_of):
		"""
		Take away the examples to conform with reward modeling input
		"""
		# Step 1: parse text formatted by the rl_dataloader
		facts= text.split("### Facts:")[-1].split("### Response:")[0]
		instruction= text.split("### Facts:")[0]
		instruction= instruction.split("Here are some examples:")[0] # if there are also the examples
		response=text.split("###Response:")[-1].split("Price Movement:")[-1]
		prompt= instruction + "\nFacts:\n"+facts + "\n\nPrice Movement:\n"+ response
		prompt=prompt.split("### Instruction:")[-1]
	
		"""
		#debug
		print('#########################')
		print('\nPost-processed prompt:')
		print(prompt)
		"""
		
		# Step 2: Prepare the input
		return tokenizer(prompt,
			truncation=True,  
			padding= padding, 
			pad_to_multiple_of=pad_to_multiple_of, 
			max_length= max_len, 
			return_tensors="pt",
			)
	
	reward_model_inputs = [parse_text_and_tokenize(text, tokenizer, padding, max_len, pad_to_multiple_of) for text in texts]
	# Compute the rewards for all texts (query + response) in the batch
	device = next(reward_model.parameters()).device
	try:
		with torch.no_grad():
			logits = [reward_model(input_ids=r['input_ids'].to(device), attention_mask=r['attention_mask'].to(device)).logits for r in reward_model_inputs]
			del reward_model_inputs
			return logits
	except Exception as e:
			print(f"Error during generation: {e}")
			return None
