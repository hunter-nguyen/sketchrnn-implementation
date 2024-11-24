import tensorflow as tf
import numpy as np
from data_loader import process_data
from utils import plot_stroke

# Process the data
file_path = 'https://storage.googleapis.com/quickdraw_dataset/full/raw/cat.ndjson'
processed_data, max_seq_length = process_data(
    file_path,
    max_samples=1000,
    max_seq_length=None
)

# Create training dataset
BATCH_SIZE = 32
dataset = tf.data.Dataset.from_tensor_slices(processed_data)
dataset = dataset.shuffle(1000).batch(BATCH_SIZE)

# Visualize a few examples to verify
for i in range(3):  # Show 3 random examples
    example = processed_data[np.random.randint(len(processed_data))]
    plot_stroke(example, f"Example {i+1}")

# Print dataset statistics
print(f"Dataset shape: {processed_data.shape}")
print(f"Max sequence length: {max_seq_length}")
print(f"Number of batches: {len(list(dataset))}")