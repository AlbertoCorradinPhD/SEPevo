SUMMARIZE_RP = """Act as a stock market expert and summarize the most recent market news. Report facts only and do not include opinions, interpretations, or analysis.\n"""

SUMMARIZE_INCIPIT ="""Execute the instructions below using English only. If the answer cannot be determined from the given information, explicitly state that the information is unavailable and do not fabricate any part of the response.
Instructions: """

SUMMARIZE_INSTRUCTION = """You are given a list of tweets about the {ticker} stock. Your task is to provide a concise summary of the key facts related to this stock, based solely on the content of the tweets. 
You shall follow these indications:
- Focus on the most significant information. Avoid unnecessary details, opinions, or speculative statements.
- Do not search the internet for additional insights. Do not perform coding. Only include what is directly mentioned in the tweets.
- Keep the summary brief and to the point. Ignore tweets that are jokes, hype, memes, personal opinions without evidence, or vague predictions.
- Do not infer, speculate, or extrapolate beyond what is explicitly stated.
- Is some themes or concrete facts are recurring, summarize them instead of reporting individual tweets. Use at most 3 bullet points or one short paragraph if possible.
- Keep the summary brief and to the point, namely strictly focused on the {ticker} stock.
Here are some examples:
{examples}
(END OF EXAMPLES)

Tweets:
{tweets}

Facts:"""

SUMMARIZE_SUFFIX  = """\nRemove from your response the following artifacts:
- every script or attempt of coding
- information that are not stricty related to the {ticker} stock
- repetitions."""

SUMMARIZE_REFINEMENT = """\nFinally, revise your response: 
- refine the reported dates (if any) 
- do not distort nor cut the {ticker} name
- ensure proper English grammar."""

SUMMARIZE_INSTRUCTION_GEMINI = """You are given a list of tweets about the a user-defined stock. Your task is to provide a concise summary of the key facts related to this stock, based solely on the content of the tweets. 
You shall follow these indications:
- Focus on the most significant information. Avoid unnecessary details, opinions, or speculative statements.
- Do not search the internet for additional insights. Do not perform coding. Only include what is directly mentioned in the tweets.
- Keep the summary brief and to the point. Ignore tweets that are jokes, hype, memes, personal opinions without evidence, or vague predictions.
- Do not infer, speculate, or extrapolate beyond what is explicitly stated.
- Is some themes or concrete facts are recurring, summarize them instead of reporting individual tweets. Use at most 3 bullet points or one short paragraph if possible.
- Keep the summary brief and to the point, namely strictly focused on the user-defined stock.
Here are some examples:
{examples}
(END OF EXAMPLES)

Facts:"""



PREDICT_INSTRUCTION = """Given a list of factual statements, assess their potential impact on the price movement of {ticker} stock. Provide your response in the following format:
(1) Price Movement: Select either 'Positive' or 'Negative'.
(2) Explanation: Offer a concise, single-paragraph explanation of your reasoning, based solely on the facts provided. Do not search for external information or incorporate any knowledge beyond the given data.
Here are some examples:
{examples}
(END OF EXAMPLES)

Facts:\n
{summary}

Price Movement:"""


PREDICT_REFLECT_INSTRUCTION = """Given a list of factual statements, assess their potential impact on the price movement of {ticker} stock. Provide your response in the following format:
(1) Price Movement: Select either 'Positive' or 'Negative'.
(2) Explanation: Offer a concise, single-paragraph explanation of your reasoning, based solely on the facts provided. Do not search for external information or incorporate any knowledge beyond the given data.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Facts:\n
{summary}

Price Movement:"""


REFLECTION_HEADER = """You have previously attempted this task and did not succeed. The following reflection(s) outline a plan to help you avoid making the same mistakes. 
Use this feedback to refine your approach and ensure you correctly address the task this time.\n"""
# See remove_reflection in explain_module.util

REFLECT_INSTRUCTION = """You are an advanced reasoning agent capable of self-improvement through reflection. After analyzing a list of facts to determine their impact on the price movement of {ticker} stock, you provided an incorrect prediction. 
In a few sentences, analyze why your assessment was incorrect, and outline a clear, high-level plan to avoid repeating this mistake in the future. Please use complete sentences.

Previous trial:
{scratchpad}

Reflection:"""

DEFAULT_INSTANCE_CAUSAL_MODEL = """Given a list of factual statements, assess their potential impact on the price movement of PFE stock. Provide your response in the following format:\n(1) Price Movement: Select either 'Positive' or 'Negative'.\n(2) Explanation: Offer a concise, single-paragraph explanation of your reasoning, based solely on the facts provided. Do not search for external information or incorporate any knowledge beyond the given data.\nHere are some examples:\nFacts:\n2016-07-26\nApple reported Q3 2016 earnings: Revenue of $42.4 billion, beating expectations. They sold 40.4 million iPhones, 9.9 million iPads, and 4.2 million Macs during that quarter.\nApple's earnings beat expectations, causing the stock to rise by almost 5% in after-hours trading.\nApple had $231.5 billion in cash reserves, enough to potentially acquire companies like Uber, Tesla, Twitter, Airbnb, Netflix, Snapchat, and SpaceX and still have billions left.\nApple's China sales were down around 29% sequentially and 33% YoY.\nDespite declining unit sales, Apple's revenue was boosted by more expensive iPad Pro models.\nApple Pay accounted for 3/4 of contactless payments in the US.\nApple's services business (App Store, Apple Music, etc.) was projected to be the size of a Fortune 500 company in the next year.\nApple was reported to be working on a car project called Project Titan, with Bob Mansfield leading it.\nThe Apple Pencil was granted a patent to work with a Mac's trackpad.\nApple faced declining iPhone sales, but the company focused on promoting apps and services.\nThe stock price experienced fluctuations after the earnings report, with significant after-hours gains.\nApple's market weight rating was reiterated by Wells Fargo, with a target price of $120.00.\n\nPrice Movement: Positive\n\nExplanation: Apple reported strong Q3 2016 earnings, surpassing revenue expectations and delivering robust sales figures across its product lines, including iPhones, iPads, and Macs. This performance exceeded market projections and triggered a nearly 5% increase in the stock's after-hours trading. Additionally, Apple's substantial cash reserves of $231.5 billion, capable of facilitating major acquisitions, demonstrated the company's financial stability and growth potential. Despite challenges in China, Apple's diverse revenue sources, including higher-priced iPad Pro models and the dominant Apple Pay in US contactless payments, contributed positively to its overall Price Movement. The promising growth trajectory of Apple's services business added further optimism. While facing declining iPhone sales, Apple's strategic focus on promoting apps and services reflected adaptability in response to changing market dynamics. The consistent support from Wells Fargo with a reiterated market weight rating and target price also reinforced investor confidence. The stock's fluctuations were notable but aligned with the positive earnings report, showcasing the market's responsiveness to Apple's performance.\n\nFacts:\n2016-04-26\nApple reported its Q2 2016 earnings, missing both profit and revenue estimates.\nApple's revenue for the quarter was $50.56 billion, falling short of the estimated $52 billion.\nThe company's adjusted earnings per share (EPS) was $1.90, lower than the expected $2.00.\nThis marks the first time in 13 years that Apple experienced a quarterly decline in revenue.\niPhone sales experienced a decline for the first time since its debut in 2007.\nThe company's guidance for the next quarter indicates expected sales of $41 billion to $43 billion.\nApple's dividend yield increased to 2.3%.\nCEO Tim Cook attributed the challenges to strong macroeconomic headwinds, especially in China.\nDespite the earnings miss, Apple announced plans to raise its dividend and return $50 billion more to shareholders.\nApple's stock price experienced a decline of around 4.8% in after-hours trading following the earnings report.\n\nPrice Movement: Negative\n\nExplanation: Apple reported disappointing Q2 2016 earnings, missing both profit and revenue estimates. The company's revenue and adjusted earnings per share fell short of expectations, marking the first quarterly revenue decline in 13 years. iPhone sales, a cornerstone of Apple's business, experienced their first-ever decline since the product's debut in 2007. The weaker-than-expected guidance for the next quarter further dampened investor Price Movement. The CEO's acknowledgment of strong macroeconomic headwinds, particularly in China, indicated external challenges affecting the company's performance. Despite announcing plans to increase dividends and return more to shareholders, the stock price plunged around 4.8% in after-hours trading following the earnings report. Overall, these factors collectively indicate a negative Price Movement surrounding AAPL stock due to its underwhelming financial performance and market outlook.\n\n(END OF EXAMPLES)\n\n\n\nFacts:\n\n2020-12-02\n\n\nHere is a summary of the key facts related to the PFE stock:\n\n• The UK government has approved the-BioNTech COVID-19 vaccine for use, and the first are expected to be delivered on December 15.\n• has signed an agreement with the Mexican government to supply 34.4 million vaccine against COVID-19.\n• The company's vaccine has been approved for use in the UK, and the first are expected to be delivered immediately.\n•'s vaccine has been shown to be effective in preventing severe illness and hospitalization due to COVID-19.\n• The company has received emergency use\n\n\n\nHere is a summary of the key facts related to the PFE stock:\n\n• The UK has approved's-19 vaccine for emergency use.\n• The vaccine is expected to be rolled out in the UK next week.\n• has received orders for 40 million of the vaccine, enough to vaccinate 20 million people.\n• The company aims to deliver 1.3 billion in 2021.\n• The vaccine has been approved in the UK before the US, with the first 800,000 expected to arrive in the country.\n• The has not yet approved the vaccine, with a meeting scheduled\n\n\n2020-12-03\n\n\nHere is a summary of the key facts related to the PFE stock:\n\n• Inc. expects to ship half of the-19 it originally planned for this year due to supply chain issues.\n• The company has been to 50 million in 2020 for a while.\n•'s Chairman and CEO stated that they are not certain whether someone can the virus after.\n• The company is working on a vaccine distribution plan.\n•'s stock price was affected by the news, with some investors experiencing losses.\n\nHere is a summary of the key facts related to the PFE\n\n\n\nPfizer $PFE expects to ship half of the-19 it originally planned for this year due to supply problems, but still plans to roll out more than a billion in 2021.\n\nPfizer $PFE expects to ship half of the-19 it originally planned for this year due to supply problems, but still plans to roll out more than a billion in 2021.\n\nPfizer $PFE expects to ship half of the-19 it originally planned for this year due to supply problems, but\n\nPrice Movement:"""
# expected negative price movement
DEFAULT_FACTS_SEQCLASS_MODEL = """\n\nFacts:2020-07-13\n\n\nHere is a summary of the key facts related to the AM stock:\n\n• AM is overbought, suggesting a potential pullback before another leg higher.\n• AM's market capitalization has increased by over $800 billion in the past four months.\n• AM's price target has been raised to $3,430 from $2,500 at.\n• AM has posted job in for leadership positions.\n• AM's stock has been mentioned as a potential top signal, along with $L and $MSFT.\n• AM is one of the top five stocks in the FAANG group\n\n\n\nHere is a summary of the key facts related to the AM stock:\n\n• Cowen raised the price target for AM from $2,750 to $3,700.\n• AM's stock price has been increasing, with some it will continue to rise.\n• There have been discussions about the potential for AM to reach new, with some it could reach $3,450.\n• AM's stock has been included in various watchlists and trading, and some have predicted it will continue to perform well.\n• There have been discussions about the potential for AM to be included in the\n\n\n\nHere is a summary of the key facts related to the AM stock:\n\n• The price target for AM was raised to $3,700 at Wedbush.\n• AM's stock has lost $120 billion in market capitalization from its morning.\n• Amazon restricts storage to prepare for the holiday shopping rush.\n• The company's CEO, Jeff Be, discussed his management and decision-making approach in 1997.\n• AM's net margin is 3.6• The company's stock is among the top 5 stocks with the highest net margin in the S 500.\n• AM\n\n\n2020-07-14\n\n\nHere is a summary of the key facts related to the AM stock:\n\n• AM has been mentioned as a stock that has been spot on in terms of its performance, with some it to reach $2,600.00.\n• AM has rolled out smart that eliminate the checkout line.\n• AM has been mentioned as one of the top companies in the world, with a market capitalization of over $1.\n• AM has been mentioned as a stock that has been heavily, with some holding large positions in the stock.\n• AM has been mentioned as a stock that has been\n\n\n\nHere is a summary of the key facts related to the AM stock:\n\n• AM broke out and tested $3065, then moved towards $3100.\n• Amazon launched new third-party health centers for employees.\n• AM's compensation philosophy is considered good.\n• The company's stock price has increased by over $800 billion in the past four months.\n• AM has a neutral fundamental analysis score.\n• The company is building health near centers.\n• AM's smart shopping cart allows customers to check out without a.\n• The company's stock price has reached a 52 high.\n• AM\n\nPrice Movement:"""
DEFAULT_RESPONSE_J = """Positive\n\nExplanation: The facts indicate a generally positive sentiment surrounding AMZN stock. Multiple price target increases by analysts, including Cowen and Wedbush, suggest strong confidence in the stock's future performance, with targets as high as $3,700. The stock has experienced significant growth, with its market capitalization increasing by over $800 billion in the past four months and reaching a 52-week high. Additionally, Amazon's innovative initiatives, such as smart shopping carts and third-party health centers for employees, highlight the company's continued focus on innovation and expansion. While there are mentions of the stock being overbought and some market capitalization loss, these are outweighed by the overall bullish outlook and strong performance indicators, leading to a positive Price Movement"""
DEFAULT_RESPONSE_K = """Negative\n\nExplanation: While AMZN (referred to as AM) has experienced significant stock price increases and positive developments, such as raised price targets, innovative product rollouts, and strong market capitalization growth, there are several cautionary signals suggesting a potential pullback. The stock is described as overbought, which often precedes a short-term decline as investors take profits. Additionally, AMZN has lost $120 billion in market capitalization during intraday trading, indicating heightened volatility and potential weakness. Mentions of the stock as a "top signal" further suggest that sentiment may be overly bullish, which can lead to a reversal. Despite the positive long-term outlook, these factors collectively point to a likely short-term negative Price Movement."""

FAULT_ANSWER = """Neutral\nExplanation: system response was unsatisfactory"""
