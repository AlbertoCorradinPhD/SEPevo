import torch

def transform_reward_to_sentiment(logit):
    """
    Transforms the output logits from the reward model to a sentiment label with score.
    
    Args:
        logit (float): The logit value from the reward model.
        
    Returns:
        dict: A dictionary containing the sentiment label and score.
    """
    
    # Apply the sign function to the logit value
    sign_value = torch.sign(torch.tensor(logit)).item()  # Convert tensor to a scalar value

    # Map sign to sentiment label and score
    if sign_value > 0:
        label = "POSITIVE"
        score = 1
    elif sign_value < 0:
        label = "NEGATIVE"
        score = -1
    else:
        label = "NEUTRAL"
        score = 0

    return {"label": label, "score": score}
