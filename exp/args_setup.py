
import os

# setup directories
def args_setup(args):
	
	args.storage_dir= os.path.join(args.root_dir, args.storage_dir)
	
	# the storage
	args.dataset_dir= os.path.join(args.root_dir, args.dataset_dir)	
	args.price_dir= os.path.join(args.dataset_dir, args.price_dir) # to keep them on storage
	args.tweet_dir= os.path.join(args.dataset_dir, args.tweet_dir)	# to keep them on storage
	
	# artifacts
	args.data_dir= os.path.join(args.run_dir, args.data_dir)		
	args.res_dir= os.path.join(args.run_dir, args.res_dir)
	
	# where intermediate artifacts are temporarily stored
	args.dataset_dir= os.path.join(args.root_dir, args.dataset_dir) # decompressed unstructured data		
	args.models_dir= os.path.join(args.root_dir, args.models_dir) # in progress checkpoints while training
		
	args.sft_adapter_path= os.path.join(args.models_dir, args.sft_adapter_path)
	args.rm_adapter_path= os.path.join(args.models_dir, args.rm_adapter_path)
	args.rl_adapter_path= os.path.join(args.models_dir, args.rl_adapter_path)
	
	
	# batch_size
	if args.GPU_RAM ==0:
		# no training
		args.sft_batch_size=0
		args.rm_batch_size=0
		args.rl_batch_size=0
		args.ppo_epochs=0
		args.early_stopping=True
		args.fewshots= False
		
	elif args.GPU_RAM <=22:
		args.sft_batch_size=4
		args.rm_batch_size=4
		args.rl_batch_size=2
		args.ppo_epochs=4
		# force RAM savings
		args.early_stopping=True
		args.fewshots= False
				
	elif args.GPU_RAM <=40: # to be tested
		args.sft_batch_size=8
		args.rm_batch_size=8
		args.rl_batch_size=4
		args.ppo_epochs=8
		args.early_stopping=False
		# force RAM savings
		args.fewshots= False
		
	else:		
		args.sft_batch_size=16
		args.rm_batch_size=16
		args.early_stopping=False
		args.ppo_epochs=16
		# force RAM savings if args.fewshots
		args.rl_batch_size=4 if args.fewshots else 8
			
	return args
