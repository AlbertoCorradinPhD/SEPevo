import os, sys
sys.path.insert(0, os.getcwd())

from utils.llm import OpenAILLM

import openai
import os

from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_sentiment(response):

    query = f"The following text provides a stock movement prediction with its explanation: {response}\n\
Task: Determine the sentiment of this prediction. Choose 'Positive' if the sentiment is favorable, \
'Negative' if the sentiment is unfavorable, or 'Neutral' if the provided text does not clearly indicate a judgment.\
Please respond with only one of the following terms: 'Positive', 'Negative', or 'Neutral'."

    
    retries = 0
    max_retries = 3
    valid_terms = ['Positive', 'Negative', 'Neutral']
    llm=OpenAILLM(model="gpt-4.1")
    
    while retries < max_retries:
        try:			
            prediction = llm(prompt=query)
            # Check if the response is valid
            if prediction.strip() in valid_terms:
                return prediction.strip()  # Return the valid response and break the loop
            else:
                print(f"Invalid response: {prediction}. Retrying...")
                retries += 1               

        except Exception as e:
            print(f"Error during execution: {e}")

    print("Max retries reached. Returning fallback sentiment.")
    # Return the first word of prediction if retries exceed limit
    return response.split()[0]


