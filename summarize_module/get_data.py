import os
import sys
sys.path.insert(0,os.getcwd() )

from data_load.dataloader import DataLoader

import pickle
import time
import glob
import pandas as pd

def get_data(args, flag="train", new=False):

	os.makedirs(args.data_dir, exist_ok=True)
	file_name = flag + '_data_summarized'

	if new:
		try:
			print("Summarizing data...")
			dataloader = DataLoader(args)
			data_summarized = dataloader.load(flag=flag)
			del dataloader
			
			if data_summarized is not None:
				print("Number of instances:", str(len(data_summarized)))
				if len(data_summarized)>0:
					print("Dumping data with pickle...")
					timestr = time.strftime("%Y%m%d-%H%M%S")
					path = os.path.join(args.data_dir, f"{file_name}_{timestr}.pkl"	)		
					with open(path, "wb") as handle:
						pickle.dump(
						data_summarized,
						handle,
						protocol=pickle.HIGHEST_PROTOCOL
						)
					print("Summarized data were saved")
				else:
					return None

		except Exception as e:
			print(f"Error summarizing tweets: {e}")
			return None

	else:
		file_list = sorted(	glob.glob(os.path.join(args.data_dir, file_name + "*.pkl"))	)
		if not file_list:
			print("No summary was found")
			return None
		try:	
			data_summarized = None
			for file_path in file_list:
				with open(file_path, "rb") as handle:
					df = pickle.load(handle)
				if not isinstance(df, pd.DataFrame):
					raise TypeError(f"{file_path} does not contain a DataFrame")

				if data_summarized is None:
					data_summarized = df
				else:
					data_summarized = pd.concat(
						[data_summarized, df],
						ignore_index=True
					)
			print("All data were loaded and merged")
			print("Number of instances:", str(len(data_summarized)))
			
		except Exception as e:
			print(f"Error: {e}")
			return None
			
	return data_summarized
