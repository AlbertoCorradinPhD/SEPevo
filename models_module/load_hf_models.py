import os, sys
sys.path.insert(0,os.getcwd() )

from utils.configurations import configurations

from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaForSequenceClassification
from transformers import BitsAndBytesConfig, GenerationConfig
from peft import prepare_model_for_kbit_training


def load_hf_model(model_path="curiousily/Llama-3-8B-Instruct-Finance-RAG"): 
	"""
	model_path=jondurbin/airoboros-l2-13b-3.1.1
	This is a fairly general purpose model, but focuses heavily on instruction following, rather than casual chat/roleplay.
	Alternatives: jondurbin/airoboros-33b-2.1, jondurbin/airoboros-65b-gpt4-2.0', ...
	"""
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	tokenizer.padding_side = "left" #for inference
	if tokenizer is None:
		print("No tokenizer for:", model_path)
	else:
		print("Tokenizer was loaded for: ", model_path)
		# debug
		#print(tokenizer)

	bnb_config, _, _, generation_kwargs= configurations()
	
				
	model = load_base_model(model_path, bnb_config)
	model.generation_config =  GenerationConfig(**generation_kwargs)
	print('New generation configuration attribute:')
	print(model.generation_config)
	model.eval()	
	
	return model, tokenizer



def load_base_model(model_path, bnb_config, device_map="auto", seq_class=False):
	
	# Load the model based on task type
	try:
		print("\nLoad base model")	
		if seq_class:
			print("A peft adapter is added to the causal model to obtain a model for sequence classification\n")
			# Common model loading parameters
			common_args = {
				"quantization_config": bnb_config,  # Config for 4-bit quantization
				"device_map": device_map,            # Automatically distribute the model across devices
				"low_cpu_mem_usage": True,           # Reduce CPU memory usage during loading
				"ignore_mismatched_sizes": True,     # Ignore size mismatch between model and checkpoint
				"num_labels": 1,                     # Single regression task (reward prediction)
				"do_sample": True                    # Enable sampling
				}

			# Check if the base model name matches "llama" or "vicuna" (case-insensitive)
			if any(model_name in model_path.lower() for model_name in ["llama", "vicuna","Vicuna"]):
				model = LlamaForSequenceClassification.from_pretrained(
					model_path,  # Path to the pre-trained model checkpoint
					**common_args  # Unpack common arguments
					)
			else:
				model = AutoModelForSequenceClassification.from_pretrained(
					model_path,  # Path to the pre-trained model checkpoint
					**common_args  # Unpack common arguments
				)

		else:	
			# Common model loading parameters
			common_args = {
				"quantization_config": bnb_config,  # Config for 4-bit quantization
				"device_map": device_map,            # Automatically distribute the model across devices
				"low_cpu_mem_usage": True,           # Reduce CPU memory usage during loading
				"do_sample": True                    # Enable sampling
				}
			if any(model_name in model_path.lower() for model_name in ["llama", "vicuna","Vicuna"]):	
				model = LlamaForCausalLM.from_pretrained(
					model_path,  # Path to the pre-trained model checkpoint
					**common_args  # Unpack common arguments
					)			
			else:
				model = AutoModelForCausalLM.from_pretrained(
					model_path,  # Path to the pre-trained model checkpoint
					**common_args  # Unpack common arguments
					)
		
		model = prepare_model_for_kbit_training(model)					
		return model
	
	except Exception as e:
		print(f"Error loading model: {e}")
		return None
