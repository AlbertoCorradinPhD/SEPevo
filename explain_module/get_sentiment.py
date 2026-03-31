import os, sys
sys.path.insert(0, os.getcwd())

from utils.llm import OpenAILLM, GeminiLLM 

import openai
import os

from tenacity import (
	retry,
	stop_after_attempt, # type: ignore
	wait_random_exponential, # type: ignore
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_sentiment(response_text, llm):
	"""
	Analyzes stock prediction text and returns a single-word sentiment label.
	"""
	
	# 1. The Prompt: We use clear instructions to force a specific output format.
	query = (
		f"The following text provides a stock movement prediction with its explanation: {response_text}\n\n"
		"Task: Determine the sentiment of this prediction. Choose 'Positive' if the sentiment is favorable, "
		"'Negative' if the sentiment is unfavorable, or 'Neutral' if the provided text does not clearly indicate a judgment. "
		"Respond with ONLY one of these terms: 'Positive', 'Negative', or 'Neutral'."
	)

	valid_terms = ['Positive', 'Negative', 'Neutral']
	
	try:			
		# 2. Execution: Calling the __call__ method of your GeminiLLM instance
		prediction = llm(query).strip()
		
		# 3. Validation: Ensure the AI didn't add extra chatter or punctuation
		if prediction in valid_terms:
			return prediction
		
		# If the AI gave a longer sentence, we try to see if a valid term is inside it
		for term in valid_terms:
			if term in prediction:
				return term

	except Exception as e:
		print(f"Error during LLM execution: {e}")
		raise # Raising the error allows the @retry decorator to catch it and try again

	# 4. Fallback: If we get here, it means the response wasn't a valid term.
	# We return 'Neutral' as a safe default for financial sentiment.
	return "Neutral"


