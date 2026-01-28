from dataclasses import dataclass
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
from typing import Union, Optional, List, Dict, Any


@dataclass
class RewardDataCollatorWithPadding(object):
	tokenizer: PreTrainedTokenizerBase
	padding: Union[bool, str, PaddingStrategy] = True
	max_len: Optional[int] = None
	pad_to_multiple_of: Optional[int] = None
	return_tensors: str = "pt"

	def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
		features_j = []
		features_k = []
		for feature in features:
			features_j.append(
				{
				"input_ids": feature["input_ids_j"],
				"attention_mask": feature["attention_mask_j"],
				}
				)
			features_k.append(
				{
				"input_ids": feature["input_ids_k"],
				"attention_mask": feature["attention_mask_k"],
				}
				)
		batch_j = self.tokenizer.pad(
			features_j,
			padding=self.padding,
			max_length=self.max_len, 
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors=self.return_tensors,
			)
		batch_k = self.tokenizer.pad(
			features_k,
			padding=self.padding,
			max_length=self.max_len, 
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors=self.return_tensors,
			)
		batch = {
			"input_ids_j": batch_j["input_ids"],
			"attention_mask_j": batch_j["attention_mask"],
			"input_ids_k": batch_k["input_ids"],
			"attention_mask_k": batch_k["attention_mask"],
			"return_loss": True,
			}
		return batch
