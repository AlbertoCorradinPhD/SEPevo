from datasets import load_dataset, DatasetDict
from datetime import datetime, timedelta
import random
import json

class RewardDataLoader(object):
	def __init__(self, data, tokenizer, val_set_size, subset_size, cutoff_len=2048, num_proc=1) -> None:

		self.data = data
		self.tokenizer = tokenizer
		self.val_set_size =  val_set_size
		self.subset_size = subset_size
		self.cutoff_len = cutoff_len
		self.num_proc = num_proc
		

    # Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
    # Then tokenize the dataset.

	def generate_and_tokenize_prompt(self, data_points):
		"""
		No labels for a reward model. Reward model must ingest the prompt, too
		"""
		new_data_points = {
			"input_ids_j": [],
			"attention_mask_j": [],
			"input_ids_k": [],
			"attention_mask_k": [],
		}
		# for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
		for question, response_j, response_k in zip(data_points["instruction"], data_points["completion_a"], data_points["completion_b"]):
			
			prompt= self._create_prompt(question)
			
			# this is different from sft because thepadding strategy is managed by the collator
			tokenized_j = self.tokenizer(
				prompt + response_j, 
				truncation=False, 
				max_length= self.cutoff_len, # to avoid default
				padding= False,
				) # padding managed by the collator
			tokenized_k = self.tokenizer(
				prompt + response_k, 
				truncation=False, 
				max_length= self.cutoff_len, # to avoid default
				padding= False,
				)
			
			new_data_points["input_ids_j"].append(tokenized_j["input_ids"])
			new_data_points["attention_mask_j"].append(tokenized_j["attention_mask"])
			new_data_points["input_ids_k"].append(tokenized_k["input_ids"])
			new_data_points["attention_mask_k"].append(tokenized_k["attention_mask"])

		return new_data_points
		
	def _create_prompt(self, question):
		"""
		Question+answer only. No incipit nor introductions
		"""		 
		facts= question.split("(END OF EXAMPLES)\n\n")[-1].split("\n\nPrice Movement:")[0].split("Facts:")[-1]
		instruction= question.split("Here are some examples:")[0]
		# no incipit: reward modeling want question+answer only
		return instruction + "\nFacts:\n"+facts +"\n\nPrice Movement:"

	def load_data(self):

		# Seed the random number generator with the current datetime
		random.seed(datetime.now().timestamp())

		# Access the 'train' split from the DatasetDict and then split it
		if self.val_set_size > 0:
			train_val = self.data["train"].train_test_split(test_size=self.val_set_size, shuffle=True)
			train_data = train_val["train"].shuffle()
			eval_data = train_val["test"].shuffle()
		else:
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
		train_data = train_data.filter(lambda x: len(x["input_ids_j"]) <= self.cutoff_len and len(x["input_ids_k"]) <= self.cutoff_len)
		
		eval_data = eval_data.map(
			self.generate_and_tokenize_prompt, batched=True, num_proc=self.num_proc, remove_columns=original_columns,
			)
		eval_data = eval_data.filter(lambda x: len(x["input_ids_j"]) <= self.cutoff_len and len(x["input_ids_k"]) <= self.cutoff_len)
		
		if len(eval_data)<10:
			return train_data, train_data
		return train_data, eval_data
