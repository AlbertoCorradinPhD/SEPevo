import os, sys
sys.path.insert(0,os.getcwd() )

from models_module.load_hf_models import load_base_model

from peft import get_peft_model, PeftModel, LoraConfig
from trl import AutoModelForCausalLMWithValueHead
# Import necessary components from accelerate
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

 
def load_weighted_adapter(peft_id, bnb_config, device_map="auto", VH=False):
	
	print("\nGoing to load peft model with weighted adapters")
	
	lora_config = LoraConfig.from_pretrained(peft_id)
	print(f"PEFT configuration loaded: {lora_config}")
	model_path= lora_config.base_model_name_or_path
	seq_class= lora_config.task_type == 'SEQ_CLS'
	base_model= load_base_model(model_path, bnb_config, device_map=device_map, seq_class=seq_class)
	
	print("\nLoad LoRA adapter")
	base_model.load_adapter(peft_id)
		
    # Apply PEFT (LoRA) config to the model
	model = PeftModel.from_pretrained(
		base_model,  # Load the base model into the PEFT model
		peft_id,  # Path to the PEFT model weights
		is_trainable=True, # default is false because it is supposed to be for inference
	)
	model.enable_adapters()
	
	# my comments
	if seq_class:
		print("Two LoRA adapters were enabled. The score layer for sequence classification plus a weighted adapter: ", peft_id)
	else: 
		print("One weighted LoRA adapter was enabled: ", peft_id)
		
	if VH:
		model = AutoModelForCausalLMWithValueHead.from_pretrained(model, 
			peft_config=lora_config, # provide lora_config for correct setup
			device_map=device_map,
			is_trainable=True,
			) 
		print("Value head was added too") 
		
	# to avoid suprises	
	for name, param in model.named_parameters():
		if 'lora' in name:
			param.requires_grad = True
			
	# Print the number of trainable parameters in the model to verify if LoRA has been applied correctly
	if VH:
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Number of trainable parameters: {trainable_params}")
	else:
		model.print_trainable_parameters()	
		
	"""
	#debug
	for name, param in model.named_parameters():
			print(name,": ",param.requires_grad)
	"""
		
	return model
       

 
def load_new_peft_model(model_path, lora_config, bnb_config, device_map="auto", seq_class=False, VH=False):
	
	print("\nGoing to load new peft model")
	
	base_model= load_base_model(model_path, bnb_config, device_map=device_map, seq_class=seq_class)
	if 	lora_config is not None:
		model = get_peft_model(base_model, peft_config=lora_config)	# adapters are already eneabled
	else:
		model= base_model
		
	# my comments
	if seq_class:
		print("Two LoRA adapters were added. The score layer for sequence classification plus new adapter.")
	else:
		print("One LoRA adapter for causal models was added.")
	
	if VH:
		model = AutoModelForCausalLMWithValueHead.from_pretrained(model, 
			peft_config=lora_config, # provide lora_config for correct setup
			device_map=device_map,
			is_trainable=True,
			)
		print("Value head was added too") 
		
	# to avoid suprises	
	for name, param in model.named_parameters():
		if 'lora' in name:
			param.requires_grad = True
	
	# Print the number of trainable parameters in the model to verify if LoRA has been applied correctly
	if VH:
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Number of trainable parameters: {trainable_params}")
	else:
		model.print_trainable_parameters()	
	
	# Now, your model is ready for training, including LoRA and low-precision setups.
	return model


