
import os
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import datetime

def summarize_trial(agents):
	correct = [a for a in agents if a.is_correct()]
	incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
	return correct, incorrect

def remove_fewshot(prompt: str) -> str:
	prefix = prompt.split('Here are some examples:')[0]
	suffix = prompt.split('(END OF EXAMPLES)')[1]
	return prefix.strip('\n').strip() + '\n\n' +  suffix.strip('\n').strip()

def remove_reflections(prompt: str) -> str:
	prefix = prompt.split('You have previously attempted this task and did not succeed.')[0] # check REFLECTION_HEADER in prompts 
	suffix = prompt.split('\n\nFacts:')[-1]
	return prefix.strip('\n').strip() + '\n\nFacts:' +  suffix.strip('\n').strip()

def log_trial(agents, trial_n):
	correct, incorrect = summarize_trial(agents)

	log = f"""
########################################
BEGIN TRIAL {trial_n}
Trial summary: Correct: {len(correct)}, Incorrect: {len(incorrect)}
#######################################
"""

	log += '------------- BEGIN CORRECT AGENTS -------------\n\n'
	for agent in correct:
		log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.target}\n\n'

	log += '------------- BEGIN INCORRECT AGENTS -----------\n\n'
	for agent in incorrect:
		log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.target}\n\n'

	return log

def save_agents(agents, res_dir, suffix='prediction', batch_number=None):
	
	output_path= os.path.join(res_dir,'agents_'+suffix)
	os.makedirs(output_path, exist_ok=True)
	try:
		for i, agent in enumerate(agents):
			if suffix=="test":
				agent.llm=None
			filename = f'{i}.joblib' if batch_number is None else f'{batch_number}{i}.joblib'
			joblib.dump(agent, os.path.join(output_path, filename))
	except Exception as e:
		print(f"Error while saving: {e}")
		

def load_agents(dir: str, isx=None, idx=None):
	"""
	Loads agents from joblib files in a specified directory.

	Args:
		dir: The directory containing the saved agent files.

	Returns:
		A list of loaded agent objects.
	"""
	agents = []
	if not os.path.exists(dir):
		print(f"Directory not found: {dir}")
		return agents

	# List files in the directory, assuming filenames are like '0.joblib', '1.joblib', etc.
	# and sort them numerically
	agent_files = sorted([f for f in os.listdir(dir) if f.endswith('.joblib')],
						 key=lambda x: int(os.path.splitext(x)[0]))
	if isx is not None and idx is not None:
		agent_files=agent_files[isx:idx]
	for filename in agent_files:
		filepath = os.path.join(dir, filename)
		try:
			agent = joblib.load(filepath)
			agents.append(agent)
		except Exception as e:
			print(f"Error loading agent from {filepath}: {e}")

	return agents
 
