import matplotlib.pyplot as plt

def plot_data(data, subject, save_path=None, minimum=False):
	"""
	Plots time course over training epochs and highlights the minimum value.
	
	Args:
	- data (list): Values to plot (one per batch/epoch).
	- subject (str): Label/title for the plot.
	- save_path (str, optional): Path to save the figure.
	"""
	# Check if the input is a valid list
	if not isinstance(data, list):
		raise ValueError("Data should be in a list")

	# Find minimum value and its index
	min_value = min(data)
	min_index = data.index(min_value) + 1  # +1 because batches start at 1

	# Create a plot
	plt.figure(figsize=(10, 6))
	plt.plot(range(1, len(data) + 1), data,
	         marker='o', color='b', label=subject)
	
	if minimum:
		# Highlight the minimum point
		plt.scatter(min_index, min_value, color='red', s=100, zorder=5, label='Minimum')

		# Annotate the minimum point
		plt.annotate(f"Min = {min_value:.4f}",
			xy=(min_index, min_value),
			xytext=(min_index, min_value * 0.75),
			arrowprops=dict(arrowstyle="->", color='red'),
			ha='center')

		# Optional: horizontal line at minimum
		plt.axhline(y=min_value, color='red', linestyle='--', alpha=0.5)

	# Add labels and title
	plt.title(subject + " during training")
	plt.xlabel("Batches")
	plt.ylabel(subject)
	plt.grid(True)

	# Add a legend
	plt.legend()

	# Save the figure if a save path is provided
	if save_path:
		plt.savefig(save_path)
		print(f"Plot saved to {save_path}")

	# Show the plot
	plt.show()




