import os, sys
sys.path.insert(0,os.getcwd() )

from toolbox.miscellaneous import create_prompt

from datetime import datetime
import random

class SFTDataLoader(object):
	def __init__(self, data, tokenizer, val_set_size, cutoff_len=2048, num_proc=1) -> None:
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
		self.cutoff_len = cutoff_len
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
			"input_ids": [],
			"attention_mask": [],
			"labels" : [],
		}
		
		for instruction, output in zip(data_points["instruction"], data_points["output"]):
			# Create the prompt based on whether there is input or not
			prompt = create_prompt(instruction)
			
			# Tokenize the full prompt with the output (for labels)
			full_tokens = self.tokenizer(
				prompt + output,
				truncation=False,  
				max_length= self.cutoff_len, # to avoid default
				padding= False,
				)

			# Calculate the length of the prompt tokens for labels	
			prompt_tokens_len = len(
				self.tokenizer(
					prompt, 
					truncation=False,  
					max_length= self.cutoff_len, 
					padding= False,
					)["input_ids"]
				)

			# Prepare labels: set labels for prompt portion to -100 (ignored by loss function)
			labels = [-100] * prompt_tokens_len + full_tokens["input_ids"][prompt_tokens_len:] #no truncation no padding							
						
			new_data_points["input_ids"].append(full_tokens["input_ids"])
			new_data_points["attention_mask"].append(full_tokens["attention_mask"])
			new_data_points["labels"].append(labels)

		return new_data_points
		

	def load_data(self):
		"""
		Loads the data, splits it into training and validation sets, and applies tokenization.

		Returns:
			tuple: A tuple containing the tokenized training data and validation data.
		
		Be careful: You are not eliminating the original content in English characters
		"""
		# Seed random number generator to ensure reproducibility
		random.seed(datetime.now().timestamp())

		# Split data into training and validation if a validation set size is specified
		if self.val_set_size > 0:
			# Split the dataset
			train_val = self.data["train"].train_test_split(test_size=self.val_set_size, shuffle=True)
			train_data = train_val["train"].shuffle()
			eval_data = train_val["test"].shuffle()
			if len(eval_data)<10:
				eval_data= train_data
		else:
			# If no validation set, just process the training data
			train_data = self.data["train"].shuffle()
			eval_data = train_data
		
		# preprocess the dataset	
		original_columns = train_data.column_names
		
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
