import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModel

# Step 1: Create a config class that includes `hidden_size` for the value head
class ValueModelConfig(PretrainedConfig):
	def __init__(self, hidden_size: int = 256, lm_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", **kwargs):
		super().__init__(**kwargs)
		self.hidden_size = hidden_size  # Further reduce the value head size
		self.lm_name = lm_name

# Step 2: Modify the ValueModel to follow PreTrainedModel conventions
class ValueModel(PreTrainedModel):
	config_class = ValueModelConfig  # Register the config class
	base_model_prefix = "llm_model"  # Set the correct prefix for the language model attribute

	def __init__(self, config: ValueModelConfig, bnb_config=None):
		super().__init__(config)
		self.config = config

		# Load the base language model (frozen) to extract features from input_ids
		self.llm_model = AutoModel.from_pretrained(
			config.lm_name,  # Use the lighter distilgpt2 model
			#quantization_config=bnb_config,  # Config for 4-bit quantization
			device_map="auto",  # Automatically distribute the model across devices
			low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
			ignore_mismatched_sizes=True,  # Ignore size mismatch between model and checkpoint
		)

		# Freeze the entire base model
		for param in self.llm_model.parameters():
			param.requires_grad = False

		# Define the value head with reduced size to save memory
		self.value_head = nn.Sequential(
			nn.Linear(self.llm_model.config.hidden_size, config.hidden_size),  # Intermediate layer with reduced size
			nn.Tanh(),
			nn.Linear(config.hidden_size, 2)  # Output a single scalar value
		)

	"""
	def forward(self, input_ids=None, attention_mask=None, **kwargs):
		# Ensure that input_ids is of type Long (torch.int64), as embedding layer requires it
		input_ids = input_ids.long() if input_ids is not None else None
		
		# Pass input through the base language model
		outputs = self.llm_model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			output_hidden_states=True,  # We need hidden states to compute the value
		)

		# Get the last hidden state (usually the last layer's output)
		last_hidden_state = outputs.last_hidden_state

		# Pool the hidden states to get a single vector per sequence.
		sequence_lengths = attention_mask.sum(dim=1) - 1
		max_len = last_hidden_state.shape[1]
		sequence_lengths = sequence_lengths.clamp(0, max_len - 1)

		# Extract the last hidden state of each sequence using attention mask
		pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), sequence_lengths]

		# Pass the pooled output through the value head
		value = self.value_head(pooled_output)
		return value
	"""
	def forward(self, input_ids=None, attention_mask=None, **kwargs):
		# Ensure that input_ids is of type Long (torch.int64), as embedding layer requires it
		input_ids = input_ids.long() if input_ids is not None else None
	
		# Pass input through the base language model without output_hidden_states
		outputs = self.llm_model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			output_hidden_states=False,  # Set to False to reduce memory usage
		)

		# Get the last hidden state (usually the last layer's output)
		last_hidden_state = outputs.last_hidden_state

		# Pool the hidden states to get a single vector per sequence.
		sequence_lengths = attention_mask.sum(dim=1) - 1
		max_len = last_hidden_state.shape[1]
		sequence_lengths = sequence_lengths.clamp(0, max_len - 1)

		# Extract the last hidden state of each sequence using attention mask
		pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), sequence_lengths]

		# Pass the pooled output through the value head
		value = self.value_head(pooled_output)
		return value


	# Implement the score() method to align with PPO's expectations
	def score(self, input_ids=None, attention_mask=None, **kwargs):
		"""Override the score method to provide the model's value prediction."""
		return self.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
