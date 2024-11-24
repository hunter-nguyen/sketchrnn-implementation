from model import SketchRNN
from data_loader import process_data
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading dataset...")
    file_path = 'https://storage.googleapis.com/quickdraw_dataset/full/raw/cat.ndjson'
    processed_data, max_seq_length = process_data(file_path, max_samples=1000)

    print(f"Data shape: {processed_data.shape}")
    print(f"Max sequence length: {max_seq_length}")

    # Create and train model
    model = SketchRNN(max_seq_length=max_seq_length)

    print("\nStarting training...")
    model.train(processed_data, epochs=10, batch_size=32)

    # Save model
    print("\nSaving model weights...")
    model.save_weights('model_weights')

    # Test generation
    print("\nTesting generation...")
    test_input = processed_data[0:1]
    generated = model.generate(test_input)
    print(f"Generated shape: {generated.shape}")
    print("Training completed successfully!")
