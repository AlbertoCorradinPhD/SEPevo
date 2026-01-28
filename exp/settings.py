import argparse

def settings():
	
	parser = argparse.ArgumentParser(description='generating')
	
	# Google Colab directories
	parser.add_argument("--root_dir", type=str, default="/content") 
	parser.add_argument("--storage_dir", type=str, default="gdrive/MyDrive", help="Storage") 
	parser.add_argument("--dataset_dir", type=str, default="sn2-main", help="Folder with unstructured or semi-structured data") 
	parser.add_argument("--program_dir", type=str, default="SEPevo", help="Folder with the software modules and its runs")
	parser.add_argument("--run_dir", type=str, default="run", help="Subfolder where artifacts will be delivered")
	parser.add_argument("--models_dir", type=str, default="saved_models") 
	
	# load data
	parser.add_argument("--price_dir", type=str, default="price/raw/") # Set to "data/sample_price/preprocessed/" for test
	parser.add_argument("--tweet_dir", type=str, default="tweet/raw/") # Set to "data/sample_tweet/raw/" for test
	parser.add_argument("--summarizer", type=str, default="curiousily/Llama-3-8B-Instruct-Finance-RAG")
	parser.add_argument("--seq_len", type=int, default=2)
	parser.add_argument("--max_stocks", type=int, default=5) 
	parser.add_argument("--max_instances_per_stock", type=int, default=300)
	parser.add_argument("--num_repetitions", type=int, default=5) 
	parser.add_argument("--num_reflect_trials", type=int, default=4)
			
	#NLP
	parser.add_argument("--cutoff_len", type=int, default=2048, help="Max length for instruction+response") # optimal input size for the model
	parser.add_argument("--max_len_sft", type=int, default=None, help="For resources optimization")
	parser.add_argument("--max_len_rm", type=int, default=None, help="For resources optimization")
	parser.add_argument("--max_len_rl", type=int, default=None, help="For resources optimization")
	parser.add_argument('--max_new_tokens', type=int, default= 256, help="Max length for response")
	parser.add_argument('--num_return_sequences', type=int, default= 1)
	parser.add_argument("--padding", type=str, default="max_length")
	parser.add_argument("--pad_to_multiple_of", type=int, default=8, help="Padding") 
	parser.add_argument("--num_proc", type=int, default=1, help="For parallel computing")
	parser.add_argument('--fewshots', type=bool, default=True, help="Add fewshots in reinforcement learning and test")	 
	
	#training parameters
	parser.add_argument("--model_path", type=str, default="lmsys/vicuna-7b-v1.5-16k") # get it from HuggingFace
	parser.add_argument('--eval_loss_threshold', type=float, default=0.1)
	parser.add_argument('--save_total_limit', type=int, default=1)
	parser.add_argument('--gradient_checkpointing', type=bool, default=True, help="Enables gradient checkpointing.") #set to True to save RAM
	parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
	parser.add_argument('--weight_decay', type=float, default=0.001) # to avoid overfitting
	parser.add_argument('--warmup_steps', type=int, default=0)
	parser.add_argument('--local_rank', type=int, default=0, help="Used for multi-gpu")
	parser.add_argument('--deepspeed', type=str, default=None, help="Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU.")
	parser.add_argument('--bf16', type=bool, default=True, help="This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU.")
	parser.add_argument("--wandb", action="store_true", default=False)
	parser.add_argument('--lr_scheduler_type', type=str, default="cosine_with_restarts") #constant
	parser.add_argument('--val_pct', type=float, default=0.2)
	parser.add_argument('--GPU_RAM', type=float, default=20, help="GPU RAM")
	
	# supervised finetuning
	parser.add_argument("--data_dir", type=str, default="data")
	parser.add_argument("--sft_adapter_path", type=str, default="sft_adapter", help="directory to save sft checkpoints")
	parser.add_argument("--ignore_data_skip", type=str, default="False")
	parser.add_argument("--sft_epochs", type=int, default=2)
	parser.add_argument('--sft_learning_rate', type=float, default=5e-4)
	parser.add_argument("--resume_from_sft_checkpoint", type=str, default='sft_adapter')
		
	# reward modeling
	parser.add_argument('--rm_learning_rate', type=float, default=5e-5)
	parser.add_argument('--reward_base_model', type=str, default="lmsys/vicuna-7b-v1.5-16k", help="The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc.")
	parser.add_argument('--rm_epochs', type=int, default=8, help="The number of training epochs for the reward model.")
	parser.add_argument('--rm_subset_size', type=int, default=1000, help="The size of the subset of the training data to use")
	parser.add_argument('--optim', type=str, default="adamw_torch", help="Enables gradient checkpointing.")
	parser.add_argument('--rm_adapter_path', type=str, default="rm_adapter", help="directory to save reward model checkpoints")
	parser.add_argument('--resume_from_rm_checkpoint', type=str, default="rm_adapter", help="If you want to resume training where it left off.") 
		
	# reinforcement learning
	parser.add_argument('--rl_learning_rate', type=float, default=1.41e-5, help="the learning rate") # smaller than usual 
	parser.add_argument('--adafactor', type=bool, default=True, help="whether to use the adafactor optimizer")
	parser.add_argument('--rl_adapter_path', type=str, default="rl_adapter", help="directory to save rl checkpoints")
	parser.add_argument('--rl_subset_size', type=int, default=1000, help="subset size for sentiment analysis")
	parser.add_argument("--rl_restarts", type=int, default=4)
	parser.add_argument('--resume_from_rl_checkpoint', type=str, default="rl_adapter", help="If you want to resume training where it left off.") 
		
	# evaluation
	parser.add_argument('--test_batch_size', type=int, default=16)
	parser.add_argument("--num_shots", type=int, default=4)
	parser.add_argument("--res_dir", type=str, default="results")
	parser.add_argument('--reward_baseline', type=float, default=0.5, help="a quality threshold")
	
	args = parser.parse_args() #parser.parse_args([]) for software development 
	return args
