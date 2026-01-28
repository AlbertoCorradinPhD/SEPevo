import evaluate
import numpy as np
import torch
	
	
def compute_accuracy(eval_pred):
	
	# Define the metric that we'll use for validation.
	accuracy = evaluate.load("accuracy")
	
	predictions, _ = eval_pred # Here, predictions is rewards_j and rewards_k.
	
	# for debug
	#calculate also the percentage of opposite signs
	# Extract predictions and rewards from eval_pred
	rewards_j, rewards_k = predictions  # Tuple: rewards_j and rewards_k
	# If the rewards are in 2D arrays, we might want to flatten them
	rewards_j = rewards_j.flatten()  # Flatten in case of multi-dimensional arrays
	rewards_k = rewards_k.flatten()  # Flatten in case of multi-dimensional arrays
	
	# Debug: Print rewards_j and rewards_k
	print("Rewards j:")
	print(rewards_j)
	print("Rewards k:")
	print(rewards_k)	
	predictions = rewards_k > rewards_j
	
	labels = np.ones(len(predictions)) # We want to see how much of the time rewards_j < rewards_k.
	acc = accuracy.compute(predictions=predictions, references=labels)
	print("Accuracy:", acc)	
	
	return acc
	
def compute_signed_metric(eval_pred):
	
	# Define the metric that we'll use for validation.
	accuracy = evaluate.load("accuracy")
	
	predictions, _ = eval_pred # Here, predictions is rewards_j and rewards_k.
	
	# Extract predictions and rewards from eval_pred
	rewards_j, rewards_k = predictions  # Tuple: rewards_j and rewards_k
	# If the rewards are in 2D arrays, we might want to flatten them
	rewards_j = rewards_j.flatten()  # Flatten in case of multi-dimensional arrays
	rewards_k = rewards_k.flatten()  # Flatten in case of multi-dimensional arrays
	
	predictions = rewards_k > rewards_j
	labels = np.ones(len(predictions)) # We want to see how much of the time rewards_j < rewards_k.
	signed_predictions= predictions*((rewards_j * rewards_k) < 0)
	signed_metric= accuracy.compute(predictions=signed_predictions, references=labels)
	signed_metric= {'Signed metric': signed_metric['accuracy']} # therefore 'eval_Signed metric' in metrics
	
	"""
	# Debug: Print rewards_j and rewards_k
	print("Rewards j:")
	print(rewards_j)
	print("Rewards k:")
	print(rewards_k)	
	print("Signed predictions:", signed_predictions)
	# Calculate the percentage of pairs with different signs
	different_signs = np.sum((rewards_j * rewards_k) < 0)  # Pairs with different signs
	percentage_diff_signs = (different_signs / len(rewards_j)) * 100  # Percentage of such pairs
	print("Percentage of pairs with different signs:", percentage_diff_signs,'%')
	print(signed_metric)
	"""
	
	return signed_metric
