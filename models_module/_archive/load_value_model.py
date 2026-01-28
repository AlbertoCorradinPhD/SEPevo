""" 
### VALUE MODEL
model_name='theminji/tinyllama-0.3-guanaco' #"TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
value_model = ValueModel(ValueModelConfig(lm_name=model_name)) 
gradient_checkpoint_setup(args,value_model)
value_model.eval()
value_model.to("cpu")
# for debug
sample_text = "This is a test sentence for the value model."
inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"] # Move inputs to the same device as the model (GPU/CPU)
attention_mask = inputs["attention_mask"] # Move inputs to the same device as the model (GPU/CPU)
with torch.no_grad():  # Disable gradient calculations
value = value_model(input_ids=input_ids, attention_mask=attention_mask)
print(f"Output Value: {value}")
print(f"Output Shape: {value.shape}")
"""
