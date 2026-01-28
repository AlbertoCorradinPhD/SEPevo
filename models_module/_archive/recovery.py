import os, sys
sys.path.insert(0,os.getcwd() )

from models_module.load_peft_models import load_weighted_adapter
from checkpoints_module.decompress_checkpoints import decompress_checkpoints

from transformers import AutoTokenizer, LlamaTokenizer, GenerationConfig
import torch
import transformers

def recovery(args, bnb_config, generation_kwargs, device_map="auto", VH=False, state_dict_path=None):
			
		# main model
		model=None
		if state_dict_path is None:
			adapter_path= args.sft_adapter_path
		else:
			adapter_path= args.rl_adapter_path
		
		done= decompress_checkpoints(
			file_path=adapter_path, 
			run_folder= args.run_dir, 
			dest_folder= args.models_dir,
			)
		if done:
			model=load_weighted_adapter(adapter_path, bnb_config, device_map, VH)
			print("Loaded peft model")
			# Set generation config
			model.generation_config =  GenerationConfig(**generation_kwargs)
			print('New generation configuration attribute:')
			print(model.generation_config)
		else:
			print("Failed to load the main model")
					
		if VH and state_dict_path is not None: 
			if os.path.exists(state_dict_path): # now that the adapter has been decompressed
				value_head_state_dict = torch.load(state_dict_path)  # Load the state_dict from file 'pytorch_model.bin'
				adjusted_value_head_state_dict = { k.replace("v_head.", ""): v for k, v in value_head_state_dict.items() if "v_head" in k}
				model.v_head.load_state_dict(adjusted_value_head_state_dict)
				print("Weighted Value Head state was loaded")
			else:
				print("Can't find weighted value Head state")			

		
		# reward model
		reward_model=None
		done= decompress_checkpoints(
			file_path=args.rm_adapter_path, 
			run_folder= args.run_dir, 
			dest_folder= args.models_dir,
			)
		if done:
			reward_model=load_weighted_adapter(args.rm_adapter_path, bnb_config, device_map)
		else:
			print("Failed to load the reward model")
			
		return model, reward_model
