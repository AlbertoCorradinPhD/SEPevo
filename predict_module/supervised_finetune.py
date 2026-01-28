import os, sys
sys.path.insert(0,os.getcwd() )

from data_load.sft_dataloader import SFTDataLoader
from data_load.sft_data_collator import SFTDataCollatorWithPadding
from toolbox.early_stopping_callback import EarlyStoppingByEvalLossCallback
from toolbox.miscellaneous import gradient_checkpoint_setup
from checkpoints_module.move_checkpoint import move_checkpoint

from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import warnings
from datasets import load_dataset
import json
import math

def supervised_finetune(args, model, tokenizer, device_map="auto", data_path=None ):
	
	if data_path is None:
		print("No data were provided. Exit")
		sys.exit	
	data = load_dataset("json", data_files=data_path)	
	dataloader = SFTDataLoader(data, 
		tokenizer, 
		val_set_size=args.val_pct, 
		cutoff_len= args.cutoff_len, 
		num_proc= args.num_proc,
		) 
	train_data, eval_data = dataloader.load_data()
	print('train data set:', str(len(train_data)))
	print(train_data)
	
	# define max_length	
	max_length=0
	for i in range(len(train_data)):
		if len(train_data[i]['input_ids'])>max_length:
			max_length=len(train_data[i]['input_ids'])
			
	if len(eval_data)!=len(train_data):
		print('eval data set:', str(len(eval_data)))
		print(eval_data)
		for i in range(len(eval_data)):
			if len(eval_data[i]['input_ids'])>max_length:
				max_length=len(eval_data[i]['input_ids'])
	else:
		print("Eval data are equal to train data")

	N= math.ceil(max_length/args.pad_to_multiple_of)
	args.max_len_sft= N*args.pad_to_multiple_of
	print("Max length for sft:", str(args.max_len_sft))
	print("Argument 'max_len_sft' was added to Namespace.")
	
	# Create an instance of your custom collator
	data_collator = SFTDataCollatorWithPadding(tokenizer=tokenizer, 
		padding= args.padding, 
		max_len= min(args.max_len_sft,args.cutoff_len), 
		pad_to_multiple_of=args.pad_to_multiple_of,
		)
	
	"""
	#debug
	for i in range(len(train_data)):
		print(len(train_data[i]['input_ids']))
	batch= data_collator(train_data)
	print(batch.keys())
	for j in range(len(batch['input_ids'])):
		print(batch['input_ids'][j].squeeze().shape)
	"""
	
	gradient_checkpoint_setup(args,model)

	training_args = TrainingArguments(		
		output_dir=args.sft_adapter_path,
		per_device_train_batch_size=args.sft_batch_size,
		per_device_eval_batch_size= args.sft_batch_size,
		warmup_steps=args.warmup_steps,
		weight_decay=args.weight_decay,
		num_train_epochs= args.sft_epochs,
		learning_rate=args.sft_learning_rate,
		fp16=False,  # Set to False to coordinate with bnb_config
		bf16=args.bf16,
		logging_strategy="steps",
		logging_steps=args.sft_batch_size,
		eval_strategy="steps",
		save_strategy="steps",
		eval_steps=args.sft_batch_size,
		save_steps=args.sft_batch_size,
		save_total_limit=args.save_total_limit,
		lr_scheduler_type=args.lr_scheduler_type,
		deepspeed=args.deepspeed,
		local_rank=args.local_rank,
		load_best_model_at_end=True,  # Ensure best model is loaded based on eval_loss
		metric_for_best_model="eval_loss",  # Specify to use eval_loss to select the best model
		greater_is_better=False,  # Since lower eval_loss is better, we set this to False
		report_to="none",
		ignore_data_skip=args.ignore_data_skip,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		gradient_checkpointing=args.gradient_checkpointing, # Enabled gradient checkpointing
		disable_tqdm=False,  # You can enable or disable tqdm for progress bars
		remove_unused_columns=True, # because data points have this format: "features: ['instruction', 'input', 'output', 'input_ids', 'labels', 'attention_mask']"
		# label_names=[], # these are the names of internal computed metrics to evaluate the model
	)

	# Create the callback instance
	early_stopping_callback = EarlyStoppingByEvalLossCallback(args.eval_loss_threshold)
	

	trainer = Trainer(
			model=model,
			train_dataset= train_data.shuffle(),
			eval_dataset= eval_data,
			args= training_args,
			data_collator=data_collator, #transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
			callbacks=[early_stopping_callback]  # Pass the callback to the Trainer
		)
	
	done=False
	print("\nTraining the model")
	with torch.autocast("cuda"): # risky mixed precision but GPU RAM savings
		try:
			trainer.train()
			print("Saving last checkpoint of the model")
			done= move_checkpoint(parent_folder=args.sft_adapter_path)		
		except Exception as e:
			print(f"Error during training: {e}")	

	return done

