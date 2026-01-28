import os, sys
sys.path.insert(0,os.getcwd() )

from checkpoints_module.move_checkpoint import move_checkpoint
from toolbox.early_stopping_callback import EarlyStoppingByEvalLossCallback_naive, EarlyStoppingByMetricCallback
from toolbox.compute_metrics import compute_accuracy, compute_signed_metric
from predict_module.reward_trainer import RewardTrainer
from data_load.rm_dataloader import RewardDataLoader
from data_load.rm_data_collator import RewardDataCollatorWithPadding
from toolbox.miscellaneous import gradient_checkpoint_setup

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import load_dataset
from transformers import TrainingArguments
import json
import math

# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "</s>"
# DEFAULT_UNK_TOKEN = "</s>"

def reward_modeling(args, model, tokenizer, device_map="auto", data_path=None):
	
	if data_path is None:
		print("No data were provided. Exit")
		sys.exit
	data = load_dataset("json", data_files=data_path)	
	reward_dataloder = RewardDataLoader(data, 
		tokenizer, 
		val_set_size=args.val_pct, 
		subset_size= args.rm_subset_size, 
		cutoff_len= args.cutoff_len, 
		num_proc= args.num_proc,
		) 
	train_data, eval_data = reward_dataloder.load_data()
	print('train data set:', str(len(train_data)))
	print(train_data)
	
	# define max_length	
	max_length=0
	for i in range(len(train_data)):
		if len(train_data[i]['input_ids_j'])>max_length:
			max_length=len(train_data[i]['input_ids_j'])
		if len(train_data[i]['input_ids_k'])>max_length:
			max_length=len(train_data[i]['input_ids_k'])
			
	if len(eval_data)!=len(train_data):
		print('eval data set:', str(len(eval_data)))
		print(eval_data)
		for j in range(len(eval_data)):
			if len(eval_data[j]['input_ids_j'])>max_length:
				max_length=len(eval_data[j]['input_ids_j'])
			if len(eval_data[j]['input_ids_k'])>max_length:
				max_length=len(eval_data[j]['input_ids_k'])
	else:
		print("Eval data are equal to train data")

	N= math.ceil(max_length/args.pad_to_multiple_of)
	args.max_len_rm= N*args.pad_to_multiple_of
	print("Max length for rm:", str(args.max_len_rm))
	print("Argument 'max_len_rm' was added to Namespace.")	
	
	# Create an instance of your custom collator
	data_collator = RewardDataCollatorWithPadding(tokenizer=tokenizer, 
		padding= args.padding, 
		max_len= min(args.max_len_rm,args.cutoff_len), 
		pad_to_multiple_of= args.pad_to_multiple_of,
		)
	
	"""
	#debug
	for i in range(len(train_data)):
		print(len(train_data[i]['input_ids_j']))
	batch= data_collator(train_data)
	batch.keys()
	for j in range(len(batch['input_ids_j'])):
		print(batch['input_ids_j'][j].squeeze().shape)
	"""
	
	gradient_checkpoint_setup(args,model)
		
	# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
	training_args = TrainingArguments(
		output_dir=args.rm_adapter_path,
		learning_rate=args.rm_learning_rate,
		per_device_train_batch_size= args.rm_batch_size, 
		per_device_eval_batch_size= args.rm_batch_size,
		num_train_epochs=args.rm_epochs,
		warmup_steps=args.warmup_steps,
		weight_decay=args.weight_decay,
		eval_strategy="steps",
		save_strategy="steps",
		eval_steps=args.rm_batch_size,
		save_steps=args.rm_batch_size,
		save_total_limit=args.save_total_limit,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		gradient_checkpointing=args.gradient_checkpointing, # Enabled gradient checkpointing
		lr_scheduler_type=args.lr_scheduler_type,
		deepspeed=args.deepspeed,
		local_rank=args.local_rank,
		fp16=False, # set to false to coordinate with bnb_config
		bf16= args.bf16,
		logging_strategy="steps",
		logging_steps=args.rm_batch_size,
		report_to="none",
		load_best_model_at_end=True,  # Ensure best model is loaded based on eval_loss
		metric_for_best_model="eval_Signed metric",  # Specify to use eval_loss to select the best model
		greater_is_better=True,  # Since lower eval_loss is better, we set this to False
		disable_tqdm=False,  # You can enable or disable tqdm for progress bars
		optim=args.optim,
		remove_unused_columns=False,
		label_names=[],
	)
	
	# Create the callback instance
	early_stopping_callback = EarlyStoppingByMetricCallback(signed_metric_threshold=1)
		
	# Train the model, woohoo.
	trainer = RewardTrainer(
		model=model,
		args=training_args,
		train_dataset= train_data.shuffle(),
		eval_dataset=eval_data,
		compute_metrics=compute_signed_metric,
		data_collator= data_collator, 
		callbacks=[early_stopping_callback]  # Pass the callback to the Trainer
	)
	  
	done=False
	print("\nTraining the model")
	with torch.autocast("cuda"): # risky mixed precision but GPU RAM savings
		try:
			trainer.train()
			print("Saving last checkpoint of the model")
			done= move_checkpoint(parent_folder=args.rm_adapter_path)
		except Exception as e:
			print(f"Error during training: {e}")	


	return done 
	
	

