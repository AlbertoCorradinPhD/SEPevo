from dataclasses import dataclass
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
from typing import Union, Optional, List, Dict, Any
import torch

@dataclass
class RLDataCollatorWithPadding(object):
	tokenizer: PreTrainedTokenizerBase
	padding: Union[bool, str, PaddingStrategy] = True
	max_len: Optional[int] = None
	pad_to_multiple_of: Optional[int] = None
	return_tensors: str = "pt"  # Set this to "pt" to indicate PyTorch tensors

	def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
		list_of_features = []

		# Add 'query' separately; we don't pad it, just keep it as-is
		queries = [feature["query"] for feature in features]

		# Process the other fields (input_ids, attention_mask)
		for feature in features:
			list_of_features.append({
				"input_ids": feature["input_ids"],
				"attention_mask": feature["attention_mask"],
			})

		# Pad input_ids and attention_mask
		batched_features = self.tokenizer.pad(
			list_of_features,
			padding=self.padding,
			max_length=self.max_len,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors=self.return_tensors,  # We use PyTorch tensors during padding
		)

		# Convert input_ids and attention_mask to lists of (unsqueezed) tensors from an embedding object
		input_ids = [ids.unsqueeze(0) for ids in batched_features["input_ids"]]
		attention_mask = [mask.unsqueeze(0) for mask in batched_features["attention_mask"]]

		# Include 'query' as part of the final batch
		batch = {
			"query": queries,  # Keep the query field as a list of strings
			"input_ids": input_ids,  # List of tensors for input_ids
			"attention_mask": attention_mask,  # List of tensors for attention_mask
			"return_loss": True,
		}

		return batch

