import os, sys
sys.path.insert(0,os.getcwd() )

from exp.args_setup import args_setup
from summarize_module.get_data import get_data 
from exp.create_datasets import create_sft_dataset, create_rm_dataset
from exp.evaluation import evaluate, gather_results 
from explain_module.get_agents import get_agents
from utils.configurations import configurations
from utils.clear_cache import clear_cache
from utils.clean_json_file import clean_json_file
from utils.fewshots import PREDICT_EXAMPLES
from models_module.get_tokenizer import get_tokenizer
from models_module.load_peft_models import load_new_peft_model,load_weighted_adapter
from models_module.verify_model_loading import verify_causal_model, verify_reward_model, verify_seqClass_model
from predict_module.supervised_finetune import supervised_finetune
from predict_module.reward_modeling import reward_modeling
from predict_module.reinforcement_learning import reinforcement_learning
from checkpoints_module.resume_checkpoints import resume_checkpoint
from checkpoints_module.compress_checkpoints import compress_checkpoints
from checkpoints_module.unload_on_storage import unload_on_storage  
from toolbox.miscellaneous import subdivide_list, count_tokens

from transformers import logging, GenerationConfig 
import torch
import transformers
from tqdm import tqdm
tqdm.pandas()

# in lack of GPU RAM
#from accelerate import Accelerator
#accelerator= Accelerator(cpu=True)


class Exp_Model:
	def __init__(self, args):
		
		args= args_setup(args)
		self.args = args				
		
		self.bnb_config, self.peft_config_sft, self.peft_config_rm, self.generation_kwargs= configurations(args)
	

	def train(self):
		
		
		##########################################################################################################
		#### SUMMARIZE CHAOTIC DATA INPUTS
		##########################################################################################################
		
		os.makedirs(self.args.run_dir, exist_ok=True)
		print('Summarize tweets step')
		for i in range(self.args.num_repetitions):
			data_summarized = get_data(self.args, new=True)
			if data_summarized is not None:
				unload_on_storage(
					run_folder=self.args.run_dir, 
					dest_folder=os.path.join(self.args.storage_dir,args.program_dir),
					)
			del data_summarized
			clear_cache()
			
		#### CREATE DATA SETS FOR MODEL TRAINING
		
		# Predictive step
		print('Predictive step')
		data_summarized= get_data(self.args)
		samples_path= os.path.join(self.args.data_dir, "samples_log.json")
		output_dir=create_sft_dataset(samples_path, self.args, self.generation_kwargs, data_summarized=data_summarized)
		if output_dir is not None:
			print("Results of predictive step were saved in folder:", output_dir)
		del data_summarized
		clear_cache()
		
		# Reflective step
		print('Self-reflection')
		comparisons_path = os.path.join(self.args.data_dir, "comparisons_log.json")
		output_dir=create_rm_dataset(samples_path, comparisons_path, self.args, self.generation_kwargs)
		if output_dir is not None:
			print("Results of reflective trials were saved in folder:", output_dir)
			
		# cleaning of data sets
		sft_data_path= os.path.join(self.args.data_dir, "sft_data.json")
		rm_data_path= os.path.join(self.args.data_dir, "rm_data.json")
		sft_data= clean_json_file(samples_path, sft_data_path)
		rm_data= clean_json_file(comparisons_path, rm_data_path)
		clear_cache()		
		unload_on_storage(
			run_folder=self.args.run_dir, 
			dest_folder=os.path.join(self.args.storage_dir,args.program_dir),
			)
			
		#### LOAD AND SETUP THE TOKENIZER
		
		device_map = "auto"
		
		tokenizer= get_tokenizer(self.args)
		if tokenizer is None:
			print("Tokenizer loading failed.")
			sys.exit()
		tokenizer.padding_side = "right" # for training
		self.args.PREDICT_EXAMPLES_NUM_TOKENS= count_tokens(PREDICT_EXAMPLES, tokenizer)
		print("Argument 'PREDICT_EXAMPLES_NUM_TOKENS' was added to Namespace.")	
		print("Number of tokens for 'PREDICT_EXAMPLES':", str(self.args.PREDICT_EXAMPLES_NUM_TOKENS))		
		
		
		##########################################################################################################
		### TRAIN SUPERVISED POLICY 
		##########################################################################################################
		
		print("Supervised Fine (Prompt) Tuning\n")
		if self.args.resume_from_sft_checkpoint:
			model= resume_checkpoint(self.args.resume_from_sft_checkpoint, self.args, self.bnb_config, device_map)		
		if model is None:
			model= load_new_peft_model(self.args.model_path, self.peft_config_sft, self.bnb_config, device_map)
			print("\nNew peft model is ready for training")
		
		done= supervised_finetune(self.args, model, tokenizer, device_map, data_path=sft_data_path)
		del model
		clear_cache()

		if done:
			print('\nCheckpoint saved at: ', self.args.sft_adapter_path,'\n' )
			sft_model=load_weighted_adapter(self.args.sft_adapter_path, self.bnb_config, device_map)		
			sft_model.eval()
			_= verify_causal_model(sft_model, tokenizer, self.args, self.generation_kwargs)
		else:
			print("Supervised Fine (Prompt) Tuning failed.")
			sys.exit()
		clear_cache()
		
		# compress checkpoints
		compress_checkpoints(
				file_path= self.args.sft_adapter_path,
				run_folder=os.path.join(self.args.root_dir, self.args.program_dir, self.args.run_dir),
				models_folder= os.path.join(self.args.root_dir, self.args.models_dir),
				)
		# save checkpoints on storage
		unload_on_storage(
			run_folder=self.args.run_dir, 
			dest_folder=os.path.join(self.args.storage_dir,args.program_dir),
			)
			
		del sft_model
		clear_cache()
		
		##########################################################################################################
		### REWARD MODELING
		##########################################################################################################
		
		print("Reward training\n")
		if self.args.resume_from_rm_checkpoint:
			model= resume_checkpoint(self.args.resume_from_rm_checkpoint, self.args, self.bnb_config, device_map)	
		if model is None:
			model= load_new_peft_model(self.args.model_path, self.peft_config_rm, self.bnb_config, device_map, seq_class=True)
			print("\nNew peft model is ready for training")
		
		print("\nTest downloaded model for rewards with opposite opinions on a movie:")
		_= verify_seqClass_model(model, tokenizer, self.args, test_input="this movie was really bad!!")	
		_= verify_seqClass_model(model, tokenizer, self.args, test_input="this movie was really good!!")
		_= verify_reward_model(model, tokenizer, self.args)	
		clear_cache()
					
				
		done= reward_modeling(self.args, model, tokenizer, data_path=rm_data_path)
		del model
		clear_cache()
		
		if done:
			print('\nCheckpoint saved at: ', self.args.rm_adapter_path,'\n' )
			reward_model=load_weighted_adapter(self.args.rm_adapter_path, self.bnb_config, device_map)	
			reward_model.eval()
			_= verify_reward_model(reward_model, tokenizer, self.args)
		else:
			print("Reward modeling failed.")
			sys.exit()
		clear_cache()
		
		
		# compress checkpoints
		compress_checkpoints(
				file_path= self.args.rm_adapter_path,
				run_folder=os.path.join(self.args.root_dir, self.args.program_dir, self.args.run_dir),
				models_folder= os.path.join(self.args.root_dir, self.args.models_dir),
				)
		# save checkpoints on storage
		unload_on_storage(
			run_folder=self.args.run_dir, 
			dest_folder=os.path.join(self.args.storage_dir,args.program_dir),
			)
		
		del reward_model
		clear_cache()
		
		del tokenizer
		clear_cache()
		
		###############################################################
		#### OPTIMIZE USING REINFORCEMENT LEARNING		
		##############################################################
		
		print("Reinforcement Learning\n")
		
		### RECOVER PREVIOUS MODELS
		model= resume_checkpoint(self.args.resume_from_sft_checkpoint, self.args, self.bnb_config, device_map, VH=True)
		if model is None:
			print("No sft model. I exit")
			sys.exit()	
		# Set generation config
		model.generation_config =  GenerationConfig(**generation_kwargs)
		print('New generation configuration attribute:')
		print(model.generation_config)
		clear_cache()
		
		reward_model = resume_checkpoint(self.args.resume_from_rm_checkpoint, self.args, self.bnb_config, device_map)
		if reward_model is None:
			print("No reward model. I exit")
			sys.exit()	
		reward_model.eval()		
		#reward_model.to("cpu") # to save GPU RAM
		clear_cache()				
	
		# Add a reference model for RL
		print("\nLoading a reference model")
		ref_model=load_new_peft_model(self.args.model_path, None, bnb_config, device_map, seq_class=False, VH=True)
		ref_model.eval()
		clear_cache()
		_= verify_causal_model(ref_model, tokenizer, self.args, self.generation_kwargs)				
		
		"""
		# debug
		### CHECK RESOURCES
		print('CUDA usage:')
		# Check memory allocated on GPU and convert to GB
		memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
		print(f'Memory allocated: {memory_allocated:.2f} GB')

		# Check memory reserved on GPU and convert to GB
		memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
		print(f'Memory reserved: {memory_reserved:.2f} GB')
		"""
		
		# REINFORCEMENT LEARNING
		tokenizer.padding_side = "left" #for inference
		done= reinforcement_learning(model, reward_model, ref_model, tokenizer, self.args, self.generation_kwargs, device_map, data_path=sft_data_path)
		del model, reward_model, ref_model
		clear_cache()
		
		if done:
			print('Checkpoint saved at: ', self.args.rl_adapter_path )
			pol_model=load_weighted_adapter(self.args.rl_adapter_path, self.bnb_config, device_map, VH=True) # I do not load the state dictionary, yet
			pol_model.eval()
			_= verify_causal_model(pol_model, tokenizer, self.args, self.generation_kwargs, VH=True)	
		else:
			print("Reinforcement Learning failed.")
			sys.exit()
		clear_cache()	
		
		# compress checkpoints
		compress_checkpoints(
				file_path= self.args.rl_adapter_path,
				run_folder=os.path.join(self.args.root_dir, self.args.program_dir, self.args.run_dir),
				models_folder= os.path.join(self.args.root_dir, self.args.models_dir),
				)
		# save checkpoints on storage
		unload_on_storage(
			run_folder=self.args.run_dir, 
			dest_folder=os.path.join(self.args.storage_dir,args.program_dir),
			)
		del pol_model
		clear_cache()	
		
		print("Train chapter is ended")
		return
		
	def test(self):
			
		##########################################################################################################
		#### SUMMARIZE CHAOTIC DATA INPUTS
		##########################################################################################################
		
		print('Summarize tweets step')
		for i in range(self.args.num_repetitions):
			data_summarized = get_data(self.args, flag="test", new=True)
			if data_summarized is not None:
				unload_on_storage(
					run_folder=self.args.run_dir, 
					dest_folder=os.path.join(self.args.storage_dir,args.program_dir),
					)
			del data_summarized
			clear_cache()
				 	
		# get agents and summarize data before loading models on the GPU RAM
		data_summarized= get_data(self.args, flag='test')
		agents= get_agents(self.args, self.generation_kwargs, suffix='test', data_summarized=data_summarized)
		print("Number of test agents: ",str(len(agents)))
		clear_cache()
			
		#### LOAD AND SETUP THE TOKENIZER
		
		device_map = "auto"
		
		tokenizer= get_tokenizer(self.args)
		if tokenizer is None:
			print("Tokenizer loading failed.")
			sys.exit()
		tokenizer.padding_side = "left" # for inference
		self.args.PREDICT_EXAMPLES_NUM_TOKENS= count_tokens(PREDICT_EXAMPLES, tokenizer)
		print("Argument 'PREDICT_EXAMPLES_NUM_TOKENS' was added to Namespace.")	
		print("Number of tokens for 'PREDICT_EXAMPLES':", str(self.args.PREDICT_EXAMPLES_NUM_TOKENS))
		tokenizer.padding_side = "left"  # for inference
					
		###############################################################
		#### TEST TRAINED LLM	
		##############################################################			
					
		### RECOVER PREVIOUS MODELS
		model= resume_checkpoint(self.args.resume_from_rl_checkpoint, self.args, self.bnb_config, device_map, VH=True)
		if model is None:
			print("No rl model. I exit")
			sys.exit()	
		# Set generation config
		model.generation_config =  GenerationConfig(**generation_kwargs)
		print('New generation configuration attribute:')
		print(model.generation_config)
		
		# Add weighted value head to main model
		state_dict_path=os.path.join(self.args.rl_adapter_path, 'pytorch_model.bin')
		if os.path.exists(state_dict_path):
			value_head_state_dict = torch.load(state_dict_path)  # Load the state_dict from file 'pytorch_model.bin'
			adjusted_value_head_state_dict = { k.replace("v_head.", ""): v for k, v in value_head_state_dict.items() if "v_head" in k}
			model.v_head.load_state_dict(adjusted_value_head_state_dict)
			print("Weighted Value Head state was loaded")
		else:
			print("Can't find weighted value Head state")	
		model.eval()	
		
		reward_model = resume_checkpoint(self.args.resume_from_rm_checkpoint, self.args, self.bnb_config, device_map)
		if reward_model is None:
			print("No reward model. I exit")
			sys.exit()
		reward_model.eval()		
		#reward_model.to("cpu") # to save GPU RAM
		clear_cache()
			
		
		### TEST		
					
		for itx, batch in tqdm(enumerate(subdivide_list(agents, self.args.test_batch_size))):
			print("Batch number: ", str(itx))
			#break
			output_dir= evaluate(batch, model, tokenizer, reward_model, self.args, self.generation_kwargs, itx)
			clear_cache()
		gather_results(output_dir)
		
		# save checkpoints on storage
		unload_on_storage(
			run_folder=self.args.run_dir, 
			dest_folder=os.path.join(self.args.storage_dir,args.program_dir),
			)
		
		print("Test chapter is ended")
		return

