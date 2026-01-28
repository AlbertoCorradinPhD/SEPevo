
from transformers import AutoTokenizer, LlamaTokenizer

def get_tokenizer(args):
	
	try:
		if any(model_name in args.model_path.lower() for model_name in ["llama", "vicuna", "Vicuna"]):
			tokenizer = LlamaTokenizer.from_pretrained(args.model_path, add_eos_token=True )
		else:
			tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
		
		# adjustments:
		tokenizer.pad_token_id = 0  # like unk: we want this to be different from the eos token
		# DEFAULT_PAD_TOKEN = 0
		# DEFAULT_EOS_TOKEN = 2
		# DEFAULT_BOS_TOKEN = 1
		# DEFAULT_UNK_TOKEN = 0
		
		print("Tokenizer was loaded")
		return tokenizer
		
	except Exception as e:
		print(f"Error loading tokenizer: {e}")
		return None


