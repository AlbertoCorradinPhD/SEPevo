from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType
import torch

def configurations(args=None):
	
		if args is not None and args.bf16:
			compute_dtype= torch.bfloat16
		else:
			compute_dtype= torch.float32

		# Recommended Parameters for 4-bit quantization
		bnb_config = BitsAndBytesConfig(
			load_in_4bit=True,  # Load the model in 4-bit precision
			bnb_4bit_compute_dtype= compute_dtype,
			bnb_4bit_quant_type="nf4", # Use the NF4 quantization type, which is optimal for the 4-bit normal float data type
			bnb_4bit_use_double_quant=True, # Use double quantization for slightly better precision
			llm_int8_threshold=6.0, # A reasonable threshold for 8-bit operations if any remain
			llm_int8_enable_fp32_cpu_offload=True, # Usually not needed with 4-bit loading			
		)
		
		peft_config_sft = LoraConfig(
			task_type=TaskType.CAUSAL_LM,
			inference_mode=False,
			r=8,
			lora_alpha=16,  # 32,
			lora_dropout=0.05,  # 0.1,
			bias="none",
			target_modules=['q_proj',  'v_proj'],
		)
		
		peft_config_rm = LoraConfig(
			task_type=TaskType.SEQ_CLS,
			inference_mode=False,
			r=8,
			lora_alpha=32, #16
			lora_dropout=0.1,  # 0.0.5,
			bias="none",
			target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
		)
		
		generation_kwargs = {
			"min_length": -1,
			'num_return_sequences': 1 if args is None else args.num_return_sequences, 
			'do_sample' : True,
			'max_new_tokens' : 128 if args is None else args.max_new_tokens, 
			'temperature' : 0.7,  # Optional: controls the randomness (0.0 -> 2.0)
			'top_p' : 0.3, #focused answer (0.0 <- 1.0),
		}
		
		return bnb_config, peft_config_sft, peft_config_rm, generation_kwargs
