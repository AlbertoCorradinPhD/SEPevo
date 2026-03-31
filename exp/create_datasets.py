import os, sys
sys.path.insert(0,os.getcwd() )

import copy
import json

from explain_module.save_results import save_results
from explain_module.get_agents import get_agents
from explain_module.util import summarize_trial, save_agents, remove_reflections
from utils.clear_cache import clear_cache

def create_sft_dataset(samples_path, args, generation_kwargs, data_summarized=None):
		
	if not os.path.exists(samples_path):
		# initialize to avoid ifelse
		with open(samples_path, 'w') as f:
			f.write(json.dumps(""))	
		
		agents= get_agents(args, generation_kwargs, suffix='prediction', data_summarized=data_summarized)
		if agents is None:
			print("No agents at all. I exit")
			sys.exit
		else:
			print("Number of agents: ",str(len(agents)))
				
		for isx, agent in enumerate(agents):
			print("Let agent predict. Agent number: ", str(isx+1),"\n")
			agent.run()
			if agent.is_correct():
				prompt = agent._build_agent_prompt()
				response = agent.scratchpad.split('Price Movement: ')[-1]
				# create the data set for sft
				sample = {"instruction": prompt, "input": "", "output": response}
				with open(samples_path, 'a') as f:
					f.write(json.dumps(sample) + "\n")
								
		correct, incorrect = summarize_trial(agents)
		print(f'Finished Trial 0, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
		output_dir= save_results(agents, args.res_dir)	
		save_agents(agents, args.res_dir)
				
		del agents
		return output_dir
	else:
		print('This step was accomplished previously')
		return None

def create_rm_dataset(samples_path, comparisons_path, args, generation_kwargs):	
			
	if not os.path.exists(comparisons_path):
		# initialize to avoid ifelse
		with open(comparisons_path, 'w') as f:
			f.write(json.dumps(""))	
		
		agents_prediction=get_agents(args, generation_kwargs)
		if agents_prediction is None:
			print("No agents at all. I exit")
			sys.exit
		else:
			print("Number of test agents: ",str(len(agents_prediction)))
			agents=copy.deepcopy(agents_prediction) # preserve train agents with deepcopy
			del agents_prediction
			clear_cache()
			
		for trial in range(args.num_reflect_trials):
			print('Trial number: ', str(trial+1))
			incorrect_agents= [a for a in agents if not a.is_correct()]
			print("Number of incorrect agents: ", len(incorrect_agents))
			for idx, agent in enumerate(incorrect_agents):
				print("Let agent with incorrect prediction reflect. Agent number: ", str(idx+1),"\n")
				# store the wrong response
				prev_response = agent.scratchpad.split('Price Movement: ')[-1]
				# re-run the agent
				agent.run()
				if agent.is_correct():
					prompt = remove_reflections(agent._build_agent_prompt())				
					response = agent.scratchpad.split('Price Movement: ')[-1]	
					
					# add to sft data set
					sample = {"instruction": prompt, "input": "", "output": response}
					with open(samples_path, 'a') as f:
						f.write(json.dumps(sample) + "\n")	
							
					# add to rm data set
					sample = {"instruction": prompt, "completion_a": prev_response, "completion_b": response}
					with open(comparisons_path, 'a') as f:
						f.write(json.dumps(sample) + "\n")	
					
			correct, incorrect = summarize_trial(agents)
			print(f'Finished Trial {trial+1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
			output_dir= save_results(agents, args.res_dir, suffix="reflection")
			
		save_agents(agents, args.res_dir, suffix='reflection')	
		del agents			
		return output_dir
	else:
		print('This step was accomplished previously')
		return None
