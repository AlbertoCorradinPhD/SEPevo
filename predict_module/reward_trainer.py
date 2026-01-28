from  transformers import Trainer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardTrainer(Trainer):
	
	# Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
	def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
		# Ensure inputs are on the correct device
		inputs_j = {k: v.to(model.device) for k, v in inputs.items() if k in ["input_ids_j", "attention_mask_j"]}
		inputs_k = {k: v.to(model.device) for k, v in inputs.items() if k in ["input_ids_k", "attention_mask_k"]}

		# Get input_ids and attention_mask
		input_ids_j = inputs_j["input_ids_j"]
		attention_mask_j = inputs_j["attention_mask_j"]
		input_ids_k = inputs_k["input_ids_k"]
		attention_mask_k = inputs_k["attention_mask_k"]

		# Explicitly cast attention_mask to model.dtype if needed (removed explicit check)
		# Note: Attention masks are typically longs, but can be cast to model.dtype for operations
		# in attention layers. We will cast them here.
		attention_mask_j = attention_mask_j.to(model.dtype)
		attention_mask_k = attention_mask_k.to(model.dtype)

		"""
		# Print device and dtype of inputs and a model parameter for debugging
		print(f"input_ids_j device: {input_ids_j.device}, dtype: {input_ids_j.dtype}")
		print(f"attention_mask_j device: {attention_mask_j.device}, dtype: {attention_mask_j.dtype}")
		print(f"input_ids_k device: {input_ids_k.device}, dtype: {input_ids_k.dtype}")
		print(f"attention_mask_k device: {attention_mask_k.device}, dtype: {attention_mask_k.dtype}")
		
		# Find a parameter in the model to check its device and dtype
		for name, param in model.named_parameters():
			if param.requires_grad:
				print(f"Model parameter '{name}' device: {param.device}, dtype: {param.dtype}")
				break # Print the first trainable parameter's device and dtype
		"""

		rewards_j = model(input_ids=input_ids_j, attention_mask=attention_mask_j).logits
		rewards_k = model(input_ids=input_ids_k, attention_mask=attention_mask_k).logits
		#print(type(rewards_j))
		
		# Original loss: minimized for k>j
		basic_loss = -nn.functional.logsigmoid(rewards_k - rewards_j).mean() 
		
		# Reward-related regularization
		loss_k_penalty = nn.functional.relu(-rewards_k).mean()  # Penalize negative rewards_k
		loss_j_penalty = nn.functional.relu(rewards_j).mean()  # Penalize positive rewards_j
		
		# Regularization to detach values. this mazimizes when vectors are opposite
		distances = F.pairwise_distance(rewards_k, rewards_j)
		sigmoid_normalized_distances = torch.sigmoid(distances)
		mean_sigmoid_normalized_distance = sigmoid_normalized_distances.mean()
		# debug
		#print(mean_sigmoid_normalized_distance)
		distance_penalty = 1 - mean_sigmoid_normalized_distance

		
		# Equal signs penalty
		rewards_j = rewards_j.flatten()  # Flatten in case of multi-dimensional arrays
		rewards_k = rewards_k.flatten()  # Flatten in case of multi-dimensional arrays
		equal_signs_penalty = torch.sum((rewards_j * rewards_k) > 0).float() / rewards_j.size(0)	

		# Sum the penalties
		equal_signs_weight = 2.0
		loss = (
			basic_loss
			+ loss_k_penalty
			+ loss_j_penalty
			+ distance_penalty
			+ equal_signs_weight * equal_signs_penalty
			)

		"""
		#for debug
		print ('\nBasic loss :', str(basic_loss.item()))
		print ('K penalty :', str(loss_k_penalty.item()))
		print ('J penalty :', str(loss_j_penalty.item()))
		print ('Distance penalty :', str(distance_penalty.item()))
		print ('Weighted equal signs penalty :', str(equal_signs_penalty.item()*equal_signs_weight))
		print ('Loss:', str(loss.item()))
		"""
		
		if return_outputs:
			return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
		return loss
