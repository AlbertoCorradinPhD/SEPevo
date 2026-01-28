import os, sys
sys.path.insert(0,os.getcwd() )

from explain_module.util import load_agents
from explain_module.agents import PredictReflectAgent

import os

	
def get_agents(args, generation_kwargs, suffix='prediction', data_summarized=None ):		
		
	if os.path.exists(os.path.join(args.res_dir,'agents_'+suffix)):
		# Collect train agents
		agents = load_agents(os.path.join(args.res_dir,'agents_'+suffix))			
		print(suffix+" agents were loaded")
		return agents

	elif data_summarized is not None:
		print("Initializing Agents...")
		agent_cls = PredictReflectAgent # this is the class, not the instance
		agents = [agent_cls(row['ticker'], row['summary'], row['target'], generation_kwargs=generation_kwargs) for _, row in data_summarized.iterrows()]
		print("Agents were initialized with data")
		return agents
		
	else:
		print("No data to initialize the agents")
		return None
