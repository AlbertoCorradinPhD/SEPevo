

import os, sys
sys.path.insert(0,os.getcwd() )

from trl import PPOConfig, PPOTrainer
from transformers import Adafactor


def get_PPOTrainer(args, model, tokenizer, ref_model, dataset=None):

	config = PPOConfig(
		task_name="optimization",
		model_name=args.model_path,
		steps= 200,
		learning_rate=args.rl_learning_rate,
		adap_kl_ctrl=True,
		init_kl_coef=0.05,  # Lower KL coefficient
		kl_penalty= 'full' if args.rl_batch_size>=8 else 'kl',
		target=6,
		horizon= 100,
		gamma=0.99,  # Lower gamma for more immediate rewards
		lam=0.95,
		cliprange=0.2,  # Reduce clip range for smaller updates
		cliprange_value=0.2,  # Match cliprange
		vf_coef=0.25,
		batch_size=args.rl_batch_size,
		forward_batch_size= 2 if args.rl_batch_size>=8 else 1,  
		mini_batch_size= 2 if args.rl_batch_size>=8 else 1,
		backward_batch_size= 2 if args.rl_batch_size>=8 else 1,
		gradient_accumulation_steps=args.gradient_accumulation_steps,  # Accumulate gradients to stabilize
		ppo_epochs= args.ppo_epochs,
		remove_unused_columns=True,
		log_with=None,
		tracker_kwargs={},
		accelerator_kwargs={},
		project_kwargs={},
		tracker_project_name='trl',
		max_grad_norm=0.5,  # Clip gradient norms
		seed=0,
		optimize_cuda_cache=False,
		early_stopping=args.early_stopping,
		target_kl=0.15,  # Lower target KL to ensure smaller updates
		push_to_hub_if_best_kwargs={},
		compare_steps=1,
		ratio_threshold=10.0,
		use_score_scaling=False,
		use_score_norm=False,
		score_clip=None,
	)

	
	print("Going to create the optimizer object\n")
	optimizer = None
	if args.adafactor:
		optimizer = Adafactor(
			params=filter(lambda p: p.requires_grad, model.parameters()), #seleziona i parametri del training
			scale_parameter= True,
			relative_step=True,
			warmup_init=True,
			#lr= args.rl_learning_rate, # decline if step is relative
		)

	print("Going to create the PPOTrainer object\n")
	#if args.resume_from_ppo_checkpoint:
		#call checkpoint function
		# Create the callback instance
		
	ppo_trainer = PPOTrainer(	
		config= config,
		model= model,
		dataset= dataset,
		tokenizer=tokenizer,
		ref_model=ref_model,
		optimizer=optimizer,
		)
	"""
	lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler. In other words, it is the optimizer to lead
	"""
	
	# safety checks
	if ppo_trainer.ref_model is None:
		ppo_trainer.ref_model=ref_model
	if ppo_trainer.optimizer is None:	
		ppo_trainer.optimizer=optimizer		
	
	ppo_trainer.is_peft_model=False # to prevent 'trl ppo_trainer' from touching the adapters

		
	return ppo_trainer
