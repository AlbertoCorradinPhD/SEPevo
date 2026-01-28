import openai
from typing import List, Tuple
import torch
import torch.nn.functional as F


class GPT4ReferenceModel:
	"""
	Black-box reference model wrapper using GPT-4 logprobs.

	WARNING:
	- Token-level alignment is NOT guaranteed
	- ref_logits are unavailable
	- Suitable only for evaluation / approximate KL
	
	#different tokenizer
	ref_model= GPT4ReferenceModel(
		tokenizer=tokenizer,
		openai_api_key=os.environ["OPENAI_API_KEY"],
		)
	"""	

	def __init__(self, tokenizer, openai_api_key: str, model: str = "gpt-4o-2024-11-20"):
		self.tokenizer = tokenizer
		self.model = model
		openai.api_key = openai_api_key

	def forward(
		self,
		queries: List[List[int]],
		responses: List[List[int]],
	) -> Tuple[torch.Tensor, None, None, None]:

		batch_logprobs = []
		lengths = []  # store sequence lengths for padding

		for query_tokens, response_tokens in zip(queries, responses):

			# Detokenize
			query_text = self.tokenizer.decode(
				query_tokens, skip_special_tokens=True
			)
			response_text = self.tokenizer.decode(
				response_tokens, skip_special_tokens=True
			)

			messages = [
				{"role": "user", "content": query_text},
				{"role": "assistant", "content": response_text},
			]

			resp = openai.chat.completions.create(
				model=self.model,
				messages=messages,
				logprobs=True,
				top_logprobs=1,
			)

			try:
				token_logprobs = resp.choices[0].logprobs.content
			except Exception as e:
				raise RuntimeError(f"Error loading or running the model: {e}")

			# 1D tensor: [seq_len]
			ref_logprobs = torch.tensor(
				[tok.top_logprobs[0].logprob for tok in token_logprobs],
				dtype=torch.float32,
			)

			batch_logprobs.append(ref_logprobs)
			lengths.append(ref_logprobs.numel())

		# --------------------------------------------------
		# Padding to max sequence length
		# --------------------------------------------------

		max_len = max(lengths)

		batch_logprobs_padded = [
			F.pad(t, (0, max_len - t.numel()), value=0.0)
			for t in batch_logprobs
		]

		# Shape: [batch_size, max_len]
		batch_logprobs_tensor = torch.stack(batch_logprobs_padded)

		return batch_logprobs_tensor, None, None, None
