import re
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from utils.prompts import SUMMARIZE_INCIPIT, SUMMARIZE_RP, SUMMARIZE_INSTRUCTION_GEMINI
from utils.fewshots import SUMMARIZE_EXAMPLES

# Initialize Vertex AI
PROJECT_ID = "serial-llm-builder-for-dt" 
LOCATION = "us-central1"           
vertexai.init(project=PROJECT_ID, location=LOCATION)

class GeminiSummarizer:
    def __init__(self, model_name="gemini-2.5-pro"):
        # Define the persona/instruction
        self.summarize_examples = SUMMARIZE_EXAMPLES
        self.summarize_prompt = SUMMARIZE_INCIPIT
        self.summarize_prompt += SUMMARIZE_INSTRUCTION_GEMINI
        self.RP = SUMMARIZE_RP
        self.system_instruction = self.RP + self.summarize_prompt
        
        # Initialize the model with the system instruction
        self.model = GenerativeModel(
            model_name=model_name,
            system_instruction=[self.system_instruction.format(
                    examples = self.summarize_examples,
                )]            
        )

    def get_token_count(self, text):
        """Calculates the number of tokens in a string using the Vertex AI API."""
        if not text:
            return 0
        try:
            response = self.model.count_tokens(text)
            return response.total_tokens
        except Exception as e:
            print(f"Error counting tokens: {e}")
            return 0

    def get_summary(self, ticker, tweets):
        if not tweets:
            return None

        # Prepare the data
        tweet_blob = "\n".join(tweets)
        user_prompt = f"Ticker: {ticker}\n\nTweets:\n{tweet_blob}\n\nFacts:"

        config = {
            "temperature": 0.7,
            "max_output_tokens": 8192,
            "top_p": 0.3,
        }

        try:
            # Generate content
            response = self.model.generate_content(
                user_prompt,
                generation_config=config,
            )
            
            summary = response.text
            
            # --- Validation Logic ---
            # 1. Calculate tokens using the new helper method
            token_count = self.get_token_count(summary)
            
            # 2. Apply filters: must be informative AND >= 1024 tokens
            if self.is_informative(summary) and token_count >= 1024:
                return summary
            else:
                # Debugging note: helps track why summaries are being discarded
                print(f"Rejected: Tokens={token_count}, Informative={self.is_informative(summary)}")
                return None
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None

    def is_informative(self, summary):
        """
        Returns False if the summary contains phrases indicating 
        lack of data or failure to generate a factual report.
        """
        if not summary:
            return False
            
        # Expanded pattern to include unavailable, insufficient, and not found
        # Using re.IGNORECASE flag for cleaner regex
        neg_patterns = [
            r"no.*information",
            r"no.*facts",
            r"no.*mention",
            r"do not contain",
            r"unavailable",
            r"insufficient",
            r"not.*found",
            r"limited.*data"
        ]
        
        # Combine patterns into a single regex pipe
        combined_regex = "|".join(neg_patterns)
        
        # re.search looks anywhere in the string; re.I makes it case-insensitive
        return not bool(re.search(combined_regex, summary, re.I))
