from transformers import TrainerCallback

class EarlyStoppingByEvalLossCallback_naive(TrainerCallback):
	def __init__(self, eval_loss_threshold: float):
		"""
		Custom callback to stop training when eval_loss is lower than a set threshold.
		
		Args:
			eval_loss_threshold (float): The threshold below which training will stop.
		"""
		self.eval_loss_threshold = eval_loss_threshold

	def on_evaluate(self, args, state, control, metrics=None, **kwargs):
		"""
		Called when evaluation happens. Checks if eval_loss is below the threshold.
		If it is, sets the `control.should_training_stop` to True, which stops training.
		
		Args:
			args (TrainingArguments): The training arguments.
			state (TrainerState): The state of the trainer.
			control (TrainerControl): A control object that can be used to modify training behavior.
			metrics (dict): Dictionary containing evaluation metrics (e.g., eval_loss).
			**kwargs: Additional keyword arguments passed by the Trainer.
		"""
		try:
			if metrics and "eval_loss" in metrics:
				eval_loss = metrics["eval_loss"]
				print(f"Evaluating... eval_loss: {eval_loss}")

				if eval_loss <= self.eval_loss_threshold:
					print(f"Stopping training: eval_loss {eval_loss} <= threshold {self.eval_loss_threshold}")
					control.should_training_stop = True
		
		except Exception as e:
				print(f"Error during callback: {e}")
		
		return control



class EarlyStoppingByEvalLossCallback(TrainerCallback):
	def __init__(self, eval_loss_threshold: float):
		"""
		Stops training when:
		  1. eval_loss falls below a given threshold, OR
		  2. eval_loss increases compared to the previous evaluation.
		"""
		self.eval_loss_threshold = eval_loss_threshold
		self.last_eval_loss = None  # Store previous eval loss

	def on_evaluate(self, args, state, control, metrics=None, **kwargs):
		
		try:
			if metrics and "eval_loss" in metrics:
				eval_loss = metrics["eval_loss"]
				print(f"Evaluating... eval_loss: {eval_loss}")

				# --- Condition 1: threshold-based stopping ---
				if eval_loss <= self.eval_loss_threshold:
					print(
						f"Stopping training: eval_loss {eval_loss} "
						f"<= threshold {self.eval_loss_threshold}"
					)
					control.should_training_stop = True

				# --- Condition 2: loss increased from last evaluation ---
				if self.last_eval_loss is not None:
					if self.last_eval_loss < 1 and eval_loss > self.last_eval_loss*103/100:
						print(
							f"Stopping training: eval_loss increased significantly"
							f"({eval_loss} > {self.last_eval_loss} plus 3%)"
						)
						control.should_training_stop = True

				# Update stored loss
				if self.last_eval_loss is None or eval_loss < self.last_eval_loss:
					self.last_eval_loss = eval_loss
				else:
					pass
		except Exception as e:
				print(f"Error during callback: {e}")
		
		return control


class EarlyStoppingByMetricCallback(TrainerCallback):
	def __init__(self, signed_metric_threshold: float):

		self.metric_threshold= signed_metric_threshold
		self.last_metric = None  # Store previous eval loss

	def on_evaluate(self, args, state, control, metrics=None, **kwargs):
		
		try:
			if metrics and 'eval_Signed metric' in metrics:
				metric = metrics['eval_Signed metric']
				print(f"Evaluating... signed metric: {metric}")
				
				# --- Condition 1: threshold-based stopping ---
				if metric >= self.metric_threshold:
					print(
						f"Stopping training: signed metric {metric} "
						f">= threshold {self.metric_threshold}"
					)
					control.should_training_stop = True

				# --- Condition 1: loss decreased from last evaluation ---
				if self.last_metric is not None:
					if self.last_metric > 0.5 and metric < self.last_metric/2:
						print(
							f"Stopping training: signed metric decreased significantly"
							f"({metric} < {self.last_metric} minus 50%)"
						)
						control.should_training_stop = True

				# Update stored loss
				if self.last_metric is None or metric > self.last_metric:
					self.last_metric = metric
				else:
					pass
		except Exception as e:
				print(f"Error during callback: {e}")
		
		return control
