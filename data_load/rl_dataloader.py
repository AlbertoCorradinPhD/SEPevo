import os, sys
sys.path.insert(0,os.getcwd() )

from toolbox.miscellaneous import create_prompt

from datetime import datetime
import random

class RLDataLoader(object):
	def __init__(self, data, tokenizer, val_set_size, subset_size, cutoff_len=2048, fewshots=False, num_proc=1) -> None:
		"""
		Initializes the SFTDataLoader instance with dataset, tokenizer, cutoff length, and validation set size.

		Args:
			data (DatasetDict): The dataset containing training data.
			cutoff_len (int): The maximum length for tokenization (truncation length).
			val_set_size (float): The fraction of the dataset to use as validation set.
			tokenizer: The tokenizer used to convert text into tokens.
		"""
		self.data = data
		self.tokenizer = tokenizer
		self.val_set_size = val_set_size
		self.subset_size = subset_size
		self.cutoff_len = cutoff_len
		self.fewshots = fewshots
		self.num_proc = num_proc
		


	def generate_and_tokenize_prompt(self, data_points):
		"""
		Generates a prompt for a given data point, tokenizes it, and prepares labels for training.

		Args:
			data_points (list of dicts): Every dict is a single data point containing "instruction", "input", and "output".

		Returns:
			dict: A dictionary containing the tokenized input IDs, labels, and attention mask, batched in lists
		"""
		
		new_data_points = {
			"query": [],
			"input_ids": [],
			"attention_mask": [],
			}
		
		for instruction in data_points["instruction"]:
			# Create the prompt based on whether there is input or not
			prompt = create_prompt(instruction, self.fewshots)
		
			"""
			# for debug
			### TEST THE LENGTH OF THE INPUT
			temp_tokens= self.tokenizer(
				prompt,# + data_point["output"], # no answer
				truncation=False
			)["input_ids"][:-1]  # Remove EOS token
			l= len(temp_tokens)
			print("number of token: ",l)
			"""
		
			# Tokenize the full prompt with the output (for labels)
			full_tokens = self.tokenizer(
				prompt,# + data_point["output"], # no answer
				truncation=False,  
				max_length= self.cutoff_len,  # to avoid default 
				padding= False,
				)  
			
			new_data_points["query"].append(prompt)
			new_data_points["input_ids"].append(full_tokens["input_ids"])
			new_data_points["attention_mask"].append(full_tokens["attention_mask"])
		
		return new_data_points


	def load_data(self):
		"""
		Loads the data, splits it into training and validation sets, and applies tokenization.

		Returns:
			tuple: A tuple containing the tokenized training data and validation data.
		
		Be careful: You are not eliminating the original content in English characters
		"""
		# Seed the random number generator with the current datetime
		random.seed(datetime.now().timestamp())
		

		# Split data into training and validation if a validation set size is specified
		if self.val_set_size > 0:
			# Split the dataset
			train_val = self.data["train"].train_test_split(test_size=self.val_set_size, shuffle=True)
			train_data = train_val["train"].shuffle()
			eval_data = train_val["test"].shuffle()
		else:
			# If no validation set, just process the training data
			train_data = self.data["train"].shuffle()
			eval_data = train_data
			
		# Reduce based on subset
		if self.subset_size > 0:
			train_data = train_data.select(range(min(len(train_data), self.subset_size)))
			eval_data = eval_data.select(range(min(len(eval_data), self.subset_size)))		
			
		# preprocess the dataset	
		original_columns = train_data.column_names
		# debug
		#print(original_columns)
			
		train_data = train_data.map(
			self.generate_and_tokenize_prompt, batched=True, num_proc=self.num_proc, remove_columns=original_columns,
			)
		train_data = train_data.filter(lambda x: len(x["input_ids"]) <= self.cutoff_len)
		
		eval_data = eval_data.map(
			self.generate_and_tokenize_prompt, batched=True, num_proc=self.num_proc, remove_columns=original_columns,
			)
		eval_data = eval_data.filter(lambda x: len(x["input_ids"]) <= self.cutoff_len)	

		if len(eval_data)<10:
			return train_data, train_data
		return train_data, eval_data

