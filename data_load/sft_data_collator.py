from dataclasses import dataclass
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
from typing import Union, Optional, List, Dict, Any

@dataclass
class SFTDataCollatorWithPadding(object):
	tokenizer: PreTrainedTokenizerBase
	padding: Union[bool, str, PaddingStrategy] = True
	max_len: Optional[int] = None
	pad_to_multiple_of: Optional[int] = None
	return_tensors: str = "pt"  # Set this to "pt" to indicate PyTorch tensors

	def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
		list_of_features = []
		
		for feature in features:
			list_of_features.append(
				{
					"input_ids": feature["input_ids"],
					"attention_mask": feature["attention_mask"],
					"labels": feature["labels"],
				}
			)

		# Pad input_ids and attention_mask without converting to tensors
		batched_features = self.tokenizer.pad(
			list_of_features,
			padding=self.padding,
			max_length=self.max_len, 
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors=None,  # Don't convert to tensors yet because labels are not automatically considered
		)

		# Manually pad the labels to match the input_ids length
		padded_labels = []
		for i, feature in enumerate(features):
			input_ids_len = len(batched_features["input_ids"][i])
			labels_len = len(feature["labels"])

			if labels_len < input_ids_len:
				padded_labels.append(feature["labels"] + [-100] * (input_ids_len - labels_len)) # padding to max_length
			else:
				padded_labels.append(feature["labels"])

		batched_features["labels"] = padded_labels

		# Return the batch as tensors if needed
		if self.return_tensors == "pt":
			import torch
			batched_features["input_ids"] = torch.tensor(batched_features["input_ids"])
			batched_features["attention_mask"] = torch.tensor(batched_features["attention_mask"])
			batched_features["labels"] = torch.tensor(batched_features["labels"])
		
		batched_features["return_loss"] = True
		return batched_features


