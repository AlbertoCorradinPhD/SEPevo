import os
import json
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score
from explain_module.util import remove_fewshot
import math

def save_results(agents, res_dir, suffix='prediction'):
	"""
	Save the results from multiple agents, calculate evaluation metrics, and store them in CSV files.

	Args:
		agents (list): List of agent objects that contain response data.
		res_dir (str): Directory path where results will be saved.
		flag (str, optional): A flag used to create a subdirectory (default 'train').
	"""
	# Output directory for saving results
	output_dir = os.path.join(res_dir, 'results_' + suffix)
	os.makedirs(output_dir, exist_ok=True)

	# Initialize a DataFrame to store results
	results = pd.DataFrame()

	# Iterate through agents to collect responses, sentiments, and explanations
	for agent in agents:
		# Append results into the DataFrame
		results = pd.concat([results, pd.DataFrame([{
			'Prompt': remove_fewshot(agent._build_agent_prompt()),
			'Prediction': agent.prediction, 
			'Target': agent.target,
			'Explanation': agent.scratchpad.split('Price Movement:')[-1],			
			}])], ignore_index=True)

	# Generate timestamp for filenames
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	results_filename = f'results_{timestamp}.csv'
	coefficients_filename = f'coefficients_{timestamp}.csv'

	# Save the results DataFrame to CSV
	results.to_csv(os.path.join(output_dir, results_filename), index=False)

	# Calculate evaluation metrics
	y_true = results['Target'].tolist()
	y_pred = results['Prediction'].tolist()


	# Define possible labels for confusion matrix: 'Positive', 'Negative', 'Neutral'
	possible_labels = ['Positive', 'Negative', 'Neutral']

	# Convert any non-'Positive' or 'Negative' predictions to 'Neutral'
	y_pred = ['Neutral' if pred not in ['Positive', 'Negative'] else pred for pred in y_pred]

	# Ensure 'y_pred' contains only the defined possible labels	
	y_pred = [pred if pred in possible_labels else 'Neutral' for pred in y_pred]
	
	# Generate the confusion matrix
	cm = confusion_matrix(y_true, y_pred, labels= possible_labels)

	# Initialize binary metrics and "Neutral" category
	tp = tn = fp = fn = neutral = 0

	# Indices for Positive, Negative, and Neutral
	pos_idx = possible_labels.index('Positive')
	neg_idx = possible_labels.index('Negative')
	neu_idx = possible_labels.index('Neutral')

	# Calculate TP, TN, FP, FN, and Neutral/Other based on confusion matrix
	tp = cm[pos_idx, pos_idx]
	tn = cm[neg_idx, neg_idx]
	fp = cm[neg_idx, pos_idx]+cm[neu_idx, pos_idx]
	fn = cm[pos_idx, neg_idx]+cm[neu_idx, neg_idx]
	predicted_neutral = sum(cm[:,neu_idx])
	neutral = sum(cm[neu_idx,:])
	missed_movements = cm[pos_idx, neu_idx]+ cm[neg_idx, neu_idx]


	# Calculate overall metrics
	sensitivity = round( tp / (tp + fn), 2) if (tp + fn) > 0 else 0
	specificity = round( tn / (tn + fp), 2) if (tn + fp) > 0 else 0
	precision = round( tp / (tp + fp), 2) if (tp + fp) > 0 else 0
	recall = sensitivity  # Recall is the same as sensitivity
	"""
	# Also neutral cases are considered here
	accuracy = accuracy_score(y_true, y_pred)  # Overall accuracy across all classes
	mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true + y_pred)) > 1 else "NA"  # MCC requires more than one unique label
	# Calculate F1 score (micro average)
	f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
	"""
	# no neutral considered
	f1_score =  round( 2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0
	accuracy = round( (tp + tn) / (tp + tn + fp + fn), 2) if (tp + tn + fp + fn) > 0 else 0
	mcc = round( (tp*tn - fp*fn) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)), 2) if (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)> 0 else "NA"

	# Create a coefficients DataFrame with metrics
	coefficients = {
		'Metric': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'Sensitivity', 'Specificity', 'Precision','Recall', 
				'F1 score', 'Accuracy', 'Matthews Correlation Coefficient', 'Predicted neutral', 'Neutral', 'Missed movements'],
		'Value': [tp, tn, fp, fn, sensitivity, specificity, precision, recall, f1_score, accuracy, mcc, predicted_neutral, neutral, missed_movements]
	}

	coefficients_df = pd.DataFrame(coefficients)

	# Save the coefficients to a CSV file
	coefficients_df.to_csv(os.path.join(output_dir, coefficients_filename), index=False)

	print(f"Results and metrics saved to {output_dir}")
	
	return output_dir
