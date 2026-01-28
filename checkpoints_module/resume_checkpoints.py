import os, sys
sys.path.insert(0,os.getcwd() )

from models_module.load_peft_models import load_weighted_adapter
from checkpoints_module.decompress_checkpoints import decompress_checkpoints

import torch
import transformers

def resume_checkpoint(file_name, args, bnb_config, device_map, VH=False):
	
	done= decompress_checkpoints(
				file_name, 
				run_folder=args.run_dir, 
                dest_folder=args.models_dir,
                )
	if done:
		file_path= os.path.join(args.models_dir, file_name)
		model=load_weighted_adapter(file_path, bnb_config, device_map, VH=VH)
		print("\nPrevious checkpoint was resumed")
		return model
	else:
		print("Previous checkpoint was not found")
		return None

		
