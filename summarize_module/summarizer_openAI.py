import os, sys
sys.path.insert(0,os.getcwd() )


from utils.llm import OpenAILLM
from utils.prompts import SUMMARIZE_INSTRUCTION
from utils.fewshots import SUMMARIZE_EXAMPLES
import tiktoken
import re

class Summarizer_openAI:
	def __init__(self):
		self.summarize_prompt = SUMMARIZE_INSTRUCTION
		self.summarize_examples = SUMMARIZE_EXAMPLES
		self.model = "gpt-4o-2024-11-20"
		self.llm = OpenAILLM(model=self.model)
		try:
			self.enc = tiktoken.encoding_for_model(self.model)
		except:
			self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

	def get_summary(self, ticker, tweets):
		summary = None
		if tweets != []:
			prompt = self.summarize_prompt.format(
									ticker = ticker,
									examples = self.summarize_examples,
									tweets = "\n".join(tweets))

			while len(self.enc.encode(prompt)) > 16385:
				tweets = tweets[:-1]
				prompt = self.summarize_prompt.format(
										ticker = ticker,
										examples = self.summarize_examples,
										tweets = "\n".join(tweets))

			summary = self.llm(prompt)		   
		return summary

	def is_informative(self, summary):
		neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
		return not re.match(neg, summary)
