import torch
import gc

def clear_cache():
	print("Clear GPU cache\n\n")
	gc.collect()
	torch.cuda.empty_cache()
