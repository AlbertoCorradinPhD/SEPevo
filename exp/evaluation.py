import os, sys
sys.path.insert(0,os.getcwd() )

from explain_module.util import summarize_trial, save_agents
from explain_module.save_results import save_results


import os
import glob
import pandas as pd
import math

def evaluate(agents, model,tokenizer, reward_model, args, generation_kwargs, batch_number):        

	for i, agent in enumerate(agents):
		print("Agent number: ", str(i+1),"(over ", str(len(agents)),"agents)")
		agent.run_n_shots(model, tokenizer, reward_model, args, generation_kwargs)
			
	correct, incorrect = summarize_trial(agents)
	print(f'Finished Trial 0, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
	output_dir= save_results(agents, args.res_dir, suffix='test')		
	save_agents(agents, args.res_dir, suffix="test", batch_number=batch_number)
	return output_dir	

def gather_results(output_dir):
	# Define the pattern to match the CSV files
	pattern = 'coefficients_*.csv'
	files = glob.glob(os.path.join(output_dir, pattern))
	
	# Check if any files were found
	if not files:
		print(f"No files found matching the pattern {pattern} in {output_dir}.")
		return
	
	print('Collected files:')
	print(files)

	# Initialize an empty DataFrame
	df = pd.DataFrame()

	# Iterate through all files and add their data to the DataFrame
	for i, file in enumerate(files, start=1):
		temp = pd.read_csv(file)
		
		# Set 'Metric' as the index if not already set
		temp.set_index('Metric', inplace=True)

		# If it's the first file, initialize the DataFrame
		if i == 1:
			df = temp[['Value']].rename(columns={'Value': f'batch_{i}'})
		else:
			# Add the 'Value' column from the current file to the DataFrame
			df[f'batch_{i}'] = temp['Value']
	
	# Ensure the required metrics are present in the DataFrame
	metrics=['True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'Sensitivity', 'Specificity', 'Precision','Recall', 
				'F1 score', 'Accuracy', 'Matthews Correlation Coefficient', 'Neutral', 'Missed movements']

	# Ensure all metrics are in the DataFrame; if missing, fill with NaN
	for metric in metrics:
		if metric not in df.index:
			df.loc[metric] = [None] * df.shape[1]

	print('Database of metrics:')
	print(df)

	# Calculate metrics if required metrics are available
	required_indices = {'True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'Neutral', 'Missed movements'}
	if required_indices.issubset(df.index):
			
		tp = df.loc['True Positives'].sum()
		tn = df.loc['True Negatives'].sum()
		fp = df.loc['False Positives'].sum()
		fn = df.loc['False Negatives'].sum()
		neutral= df.loc['Neutral'].sum()
		missed_movements = df.loc['Missed movements'].sum()

		sensitivity = round( tp / (tp + fn), 2) if (tp + fn) > 0 else 0
		specificity = round( tn / (tn + fp), 2) if (tn + fp) > 0 else 0
		precision = round( tp / (tp + fp), 2) if (tp + fp) > 0 else 0
		recall = sensitivity  # Recall is the same as sensitivity
		f1_score = round( 2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0
		accuracy = round( (tp + tn) / (tp + tn + fp + fn), 2) if (tp + tn + fp + fn) > 0 else 0
		mcc = round( (tp*tn - fp*fn) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)), 2) if (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)> 0 else "NA"
		
		# Prepare results as a DataFrame
		coefficients = {
			'Metric': metrics,
			'Value': [tp, tn, fp, fn, sensitivity, specificity, precision, recall, f1_score, accuracy, mcc, neutral, missed_movements]
		}
		
		coefficients_df = pd.DataFrame(coefficients)
		coefficients_filename = 'overall_results.csv'
		coefficients_df.to_csv(os.path.join(output_dir, coefficients_filename), index=False)
		
		print("\nResults were gathered and saved to 'overall_results.csv'.")
	else:
		print("Missing required metrics (True Positives, True Negatives, etc.). Cannot calculate performance metrics.")

