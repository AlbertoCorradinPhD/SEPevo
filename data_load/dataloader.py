import os, sys
sys.path.insert(0,os.getcwd() )

from summarize_module.summarizer_openAI import Summarizer_openAI
from summarize_module.summarizer_hf import Summarizer_hf


import os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import time

class DataLoader:
	def __init__(self, args):
		self.price_dir = args.price_dir
		self.tweet_dir = args.tweet_dir
		self.seq_len = args.seq_len
		self.summarizer = Summarizer_openAI() if args.summarizer is None else Summarizer_hf(model_path=args.summarizer)
		self.max_stocks= args.max_stocks
		self.data_dir= args.data_dir
		self.max_instances_per_stock= args.max_instances_per_stock
		self.val_pct= args.val_pct


	def daterange(self, start_date, end_date):
		for n in range(int((end_date - start_date).days)):
			yield start_date + timedelta(n)


	def get_target(self, end_date_str, start_date_str, price_data): 
		# Get the value for the end date and start date, convert them to float, and calculate the difference
		end_value = float(price_data[price_data[:, 0] == end_date_str][0, 1])
		start_value = float(price_data[price_data[:, 0] == start_date_str][0, 1])

		# Now calculate the difference
		price_chg = (end_value - start_value)/start_value*100
		# it considers the open value

		if price_chg > 1:
			target = "Positive"
		elif price_chg < -1:
			target = "Negative"
		else:
			target = "Neutral"
		return target


	def get_tweets(self, ticker, date_str):
		tweets = []
		tweet_path = os.path.join(self.tweet_dir, ticker, date_str)

		if os.path.exists(tweet_path):
			with open(tweet_path) as f:
				lines = f.readlines()
				for line in lines:
					tweet_obj = json.loads(line)
					tweets.append(tweet_obj['text'])
		return tweets


	def load(self, flag='train'):
		data = pd.DataFrame()

		all_files = os.listdir(self.price_dir)
		# Sample items
		if len(all_files) > self.max_stocks:
			# Seed the random number generator with the current datetime
			random.seed(datetime.now().timestamp())
			files_to_process = random.sample(all_files, self.max_stocks)			
		else:
			files_to_process = all_files
		
		timestr = time.strftime("%Y%m%d-%H%M%S")
		file_path=os.path.join(self.data_dir,flag+'_stocks_'+timestr+'.txt')
		with open(file_path, 'w') as f:
			for file_name in files_to_process:
				f.write(f"{file_name}\n")

		# Resume original loop
		for file_name in files_to_process:
			print('\n\nFile under investigation: ', file_name)
			price_path=os.path.join(self.price_dir,file_name)
			price_data = pd.read_csv(price_path)
			price_data_sorted = price_data.sort_values(by=price_data.columns[0])
			ordered_price_data  = price_data_sorted.values
			ticker = file_name[:-4]

			tes_idx = round(len(ordered_price_data) * 0.8)
			end_idx = len(ordered_price_data)

			if flag == "test":
				data_range = range(tes_idx, end_idx)
				max_samples = int(self.max_instances_per_stock * self.val_pct)
			else:
				data_range = range(self.seq_len, tes_idx)
				max_samples = self.max_instances_per_stock
				
			range_len = len(data_range)
			if range_len > max_samples:
				sampled_idx = random.sample(data_range, max_samples)
			else:
				sampled_idx = list(data_range)

		
			for i,idx in enumerate(sampled_idx):
				print("\nTicker: ",ticker,"instance", str(i+1), "over", len(sampled_idx))					
				end_date_str = ordered_price_data[idx, 0]
				start_date_str = ordered_price_data[idx-self.seq_len, 0]
				end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
				start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
				target = self.get_target(end_date_str, start_date_str, price_data=ordered_price_data)
				# Discard neutral cases for training
				if target == "Neutral":
					# debug
					print("Discarded because target is neutral")
					continue
				
				summary_all = ""
				for seq_date in self.daterange(start_date, end_date):
					seq_date_str = seq_date.strftime("%Y-%m-%d")
					#print(seq_date_str)	
					tweet_data = self.get_tweets(ticker, seq_date_str)
					summary = self.summarizer.get_summary(ticker, tweet_data)
					if summary and summary is not None and summary != "" and self.summarizer.is_informative(summary):
						summary_all += seq_date_str + "\n" + summary + "\n\n"

				if summary_all != "":
					data = pd.concat([data, pd.DataFrame([{'ticker': ticker, 'summary': summary_all.rstrip(), 'target': target}])], ignore_index=True)
					#debug
					print('\nHere are summarized tweets:')
					print(summary_all)
				else:
					print("Unsatisfactory tweet data")

		return data
