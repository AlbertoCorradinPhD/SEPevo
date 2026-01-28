import os, sys
sys.path.insert(0,os.getcwd() )

from utils.prompts import SUMMARIZE_INSTRUCTION, SUMMARIZE_INCIPIT, SUMMARIZE_RP, SUMMARIZE_SUFFIX, SUMMARIZE_REFINEMENT
from utils.fewshots import SUMMARIZE_EXAMPLES
from models_module.load_hf_models import load_hf_model
from toolbox.generate_retry import generate_retry
from toolbox.miscellaneous import subdivide_list
from utils.clear_cache import clear_cache 

import math
import re
from tqdm import tqdm
tqdm.pandas()
import warnings

# Suppress specific UserWarning from transformers library
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.tokenization_utils_base")

class Summarizer_hf(object):
	"""
	self.optimal_tokens depends on the HF model that is used to summarize unstructured data.
	This amount shall be selected specifically, and independently from the LLM whose LoRA layers are going to be trained 
	"""	
	def __init__(self, model_path=None):
		self.summarize_prompt= SUMMARIZE_INCIPIT
		self.summarize_prompt += SUMMARIZE_INSTRUCTION.split("Here are some examples")[0]+SUMMARIZE_SUFFIX+SUMMARIZE_REFINEMENT
		self.summarize_prompt += SUMMARIZE_INSTRUCTION.split("(END OF EXAMPLES)")[-1]		
		self.summarize_examples = "" #SUMMARIZE_EXAMPLES
		self.RP= SUMMARIZE_RP
		self.model_path = model_path		
		self.model, self.tokenizer= (None, None) if self.model_path is None else load_hf_model(self.model_path)
		if self.model is None or self.tokenizer is None:
			return None
		self.min_tokens= 1024 # otherwise garbage
		self.optimal_tokens = 2048 if any(t in model_path.lower() for t in ('8b', '7b', '13b')) else 4096
		self.max_tokens= int(self.optimal_tokens*5)
		# debug
		#print(self.summarize_prompt)
	
	
	def tokenization(self, ticker, tweets, tokenizer):
		prompt = self.summarize_prompt.format(
			ticker = ticker,
			examples = self.summarize_examples,
			tweets = "\n".join(tweets)
			)
		if 'airoboros' in self.model_path:
			query =f"""[INST] <<SYS>>{self.RP}<</SYS>> {prompt}[/INST]"""
			#print("Query in llama-2 chat format")
		elif 'curiousily' in self.model_path:
			#print("Query in the 'Instruct' prompt format")
			query =f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{self.RP}<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
		else:
			query=prompt
		# debug
		#print(query)

		tokenized = tokenizer(
			query,
			truncation=False,  # Don't truncate, keep the full sequence if shorter than max_length
			max_length= 16385, #to avoid default
			padding=True,
			pad_to_multiple_of=8,
			return_tensors='pt',				
			)
		num_tokens = tokenized["input_ids"].shape[1]
		return query, tokenized, num_tokens			
	
	def get_summary(self, ticker, tweets):
			
		if tweets is not None and len(tweets)>0:			
			query, tokenized, num_tokens= self.tokenization(ticker, tweets, self.tokenizer)
			
			if self.optimal_tokens <= num_tokens <= self.max_tokens:
				batch = {
					'query': [],
					'input_ids': [], 
					'attention_mask': [],
					'num_tokens': [],
					}
				N= math.ceil(num_tokens/self.optimal_tokens)
				batch_size= int(len(tweets)/N)
				#print(batch_size)			
				for itx, tweets_batch in tqdm(enumerate(subdivide_list(tweets, batch_size))):
					#print("\nTweet batch number: ", str(itx))			
					query, tokenized, num_tokens= self.tokenization(ticker, tweets_batch, self.tokenizer)
					if num_tokens> self.min_tokens: # significant tweets
						batch['query'].append(query)
						batch['input_ids'].append(tokenized['input_ids'])
						batch['attention_mask'].append(tokenized['attention_mask'])
						batch['num_tokens'].append(num_tokens)
								
				response_tensors = generate_retry(self.model, self.tokenizer, batch)
				if response_tensors is None:
					return None			

				batch["response"]=[self.tokenizer.decode(response, skip_special_tokens=True)+"\n" for response in response_tensors]					
				summary = "\n".join(batch["response"]) 
				if len(summary)>512:
					return summary
			elif num_tokens > self.max_tokens:
				print("Tokens' cardinality is out of range:", str(num_tokens))
			else:
				print("Tokens' cardinality is poor.")	
		
		return None

	def is_informative(self, summary):
		neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
		return not re.match(neg, summary)
