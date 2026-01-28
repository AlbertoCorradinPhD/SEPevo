import os, sys
sys.path.insert(0,os.getcwd() )

from data_load.rl_dataloader import RLDataLoader
from data_load.rl_data_collator import RLDataCollatorWithPadding
from predict_module.get_PPOTrainer import get_PPOTrainer
from predict_module.rl_trainer import rl_trainer
from toolbox.miscellaneous import gradient_checkpoint_setup, batching, safe_update
from toolbox.plot_data import plot_data
from utils.clear_cache import clear_cache
from checkpoints_module.move_checkpoint import move_checkpoint
from checkpoints_module.remove_checkpoint import remove_checkpoint
from checkpoints_module.save_trainer import save_trainer
from utils.clean_json_file import clean_json_file


import torch
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
import json
import math
from tqdm import tqdm
tqdm.pandas()
import warnings

# Suppress specific UserWarning from transformers library
warnings.filterwarnings("ignore", category=UserWarning, module="trl.trainer.ppo_config")
warnings.filterwarnings("ignore", category=UserWarning, module="trl.trainer.ppo_trainer")

def reinforcement_learning(model, reward_model, ref_model, tokenizer, args, generation_kwargs, device_map="auto", data_path=None):
	
	if data_path is None:
		print("No data were provided. Exit")
		sys.exit

	data = load_dataset("json", data_files=data_path)	
	dataloader = RLDataLoader(data, 
		tokenizer, 
		val_set_size=0, 
		subset_size= args.rl_subset_size, 
		cutoff_len= args.cutoff_len+args.PREDICT_EXAMPLES_NUM_TOKENS if args.fewshots else args.cutoff_len, 
		fewshots= args.fewshots, 
		num_proc= args.num_proc,
		) 
	train_data, eval_data = dataloader.load_data()
	print('train data set:', str(len(train_data)))
	print(train_data)
	
	# No eval data set is going to be used	
	del eval_data
	clear_cache()
	
	# define max_length
	max_length=0
	for i in range(len(train_data)):
		if len(train_data[i]['input_ids'])>max_length:
			max_length=len(train_data[i]['input_ids'])
	
	N= math.ceil(max_length/args.pad_to_multiple_of)
	args.max_len_rl= N*args.pad_to_multiple_of
	print("Max length for rl:", str(args.max_len_rl))
	print("Argument 'max_len_rl' was added to Namespace.")	
	
	# Create an instance of your custom collator
	data_collator = RLDataCollatorWithPadding(tokenizer=tokenizer, 
		padding= args.padding, 
		max_len= min(args.max_len_rl,args.cutoff_len+args.PREDICT_EXAMPLES_NUM_TOKENS) if args.fewshots else min(args.max_len_rl,args.cutoff_len), 
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
		print(batch['attention_mask'][j].squeeze().shape)
	"""
	gradient_checkpoint_setup(args, model)
	trainer= get_PPOTrainer(args, model, tokenizer, ref_model, dataset=None) #data and rewards are managed outside the PPOTrainer
	# in lack of GPU RAM
	#pol_model, ref_model, reward_model, sft_model, trainer = trainer.accelerator.prepare(pol_model, ref_model, reward_model, sft_model, trainer)
	
	# CHECK THE DISTRIBUTION OF RESOURCES
	model_device=next(model.parameters()).device
	reward_model_device=next(reward_model.parameters()).device	
	ref_model_device=next(ref_model.parameters()).device
	print("Devices in use:")
	print('policy model device:', model_device)
	print('reward model device:', reward_model_device)
	print('ref model device:', ref_model_device)
	print('trainer device:', trainer.accelerator.device)
	print("Trainer configuration:")
	print(trainer.config)
	print("")


	kl_divergences = []
	loss_values = []		
	os.makedirs(args.rl_adapter_path, exist_ok=True)
	# initialize to avoid ifelse
	log_path= os.path.join(args.res_dir, "stats_log.json")
	with open(log_path, 'w') as f:
		f.write(json.dumps(""))
	done= False
	ref_loss=100
	
	print("\nTraining the model")	
	try:
		for i in range(args.rl_restarts):
			data_dict = data_collator(train_data.shuffle())
			for itx, batch in tqdm(enumerate(batching(data_dict, batch_size=args.rl_batch_size))):
				print("")
				print("Epoch: ", str(i+1),", batch number: ", str(itx+1))
				# debug
				#break
				stats, loss, kl = rl_trainer(trainer, reward_model, tokenizer, batch, args, generation_kwargs, device_map)
				loss_values, kl_divergences, flag= safe_update(stats, loss, kl, log_path, loss_values, kl_divergences)
				if not flag:
					continue
				if len(loss_values)>0 and loss_values[-1]< ref_loss: # save best
					ref_loss=loss_values[-1]
					_ = remove_checkpoint(parent_folder=args.rl_adapter_path) # remove previous checkpoints. Only one will remain policy
					output_dir = os.path.join(args.rl_adapter_path, 'checkpoint-' + str(itx+1))
					save_trainer(trainer, output_dir)
					clear_cache()
				if len(loss_values)>0 and loss_values[-1]< args.eval_loss_threshold:
					print("Target value loss was reached")
					break
			
			# end of this epoch. Repeat condition for double break
			if len(loss_values)>0 and loss_values[-1]< args.eval_loss_threshold:
				print("Stop training")
				break
				
		# after training		
		print("Saving last checkpoint of the model")
		done = move_checkpoint(args.rl_adapter_path) # save on parent folder
										
	except Exception as e:
		print(f"Error during training: {e}")	
	
	del trainer
	stats_path= os.path.join(args.res_dir, "stats_data.json")
	stats= clean_json_file(log_path, stats_path)
	try:			
		output_path=os.path.join(args.res_dir,'kl_divergences.jpeg')
		plot_data(kl_divergences, 'KL divergences',output_path)	
		output_path=os.path.join(args.res_dir,'loss_values.jpeg')
		plot_data(loss_values, 'Loss values',output_path, minimum=True)
	except Exception as e:
		print(f"Error while drawing: {e}")
			
	return done

