import tensorflow as tf
import numpy as np
from tqdm import tqdm  # For progress bars

class SketchRNN:
    def __init__(self, max_seq_length=1000):
        self.latent_dim = 128
        self.max_seq_length = max_seq_length
        self.encoder = None
        self.decoder = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_encoder(self, input_shape):
        """Simplified encoder"""
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LSTM(256),
            tf.keras.layers.Dense(self.latent_dim)
        ])

    def build_decoder(self, seq_length):
        """Simplified decoder"""
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
            tf.keras.layers.RepeatVector(seq_length),
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.Dense(3)
        ])

    @tf.function
    def train_step(self, batch):
        """Single training step"""
        with tf.GradientTape() as tape:
            # Forward pass
            encoded = self.encoder(batch)
            decoded = self.decoder(encoded)
            # Loss calculation
            loss = tf.reduce_mean(tf.square(batch - decoded))

        # Backpropagation
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def train(self, data, epochs=10, batch_size=32):
        """Train with progress bars"""
        print("Initializing training...")

        # Get shapes
        seq_length = data.shape[1]
        input_shape = (seq_length, 3)

        # Build models if not built
        if self.encoder is None:
            print("Building encoder...")
            self.encoder = self.build_encoder(input_shape)
        if self.decoder is None:
            print("Building decoder...")
            self.decoder = self.build_decoder(seq_length)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
        num_batches = len(list(dataset))

        print(f"Training on {len(data)} samples with {num_batches} batches per epoch")

        # Training loop with progress bar
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0

            # Progress bar for batches
            with tqdm(total=num_batches, desc=f"Epoch {epoch+1}") as pbar:
                for batch in dataset:
                    loss = self.train_step(batch)
                    epoch_loss += loss
                    pbar.update(1)
                    pbar.set_postfix({'loss': float(loss)})

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    def generate(self, input_strokes, temperature=1.0):
        """Generate continuation of input strokes"""
        if self.encoder is None or self.decoder is None:
            raise Exception("Model not initialized. Train the model first.")

        # Get sequence length from input
        seq_length = input_strokes.shape[0]

        # Rebuild decoder if needed
        if self.decoder.output_shape[1] != seq_length:
            self.decoder = self.build_decoder(seq_length)

        # Encode input
        encoded = self.encoder(input_strokes[np.newaxis, ...])

        # Generate from decoder
        decoded = self.decoder(encoded)

        # Apply temperature
        decoded = decoded / temperature

        return decoded[0].numpy()  # Remove batch dimension

    def save_weights(self, filepath):
        """Save model weights"""
        if self.encoder is None or self.decoder is None:
            raise Exception("Model not initialized. Train the model first.")
        self.encoder.save_weights(filepath + '_encoder')
        self.decoder.save_weights(filepath + '_decoder')

    def load_weights(self, filepath):
        """Load model weights"""
        try:
            # Get sequence length from saved encoder weights
            if self.encoder is None or self.decoder is None:
                # Try to determine shapes from saved weights
                temp_data = np.zeros((1, self.max_seq_length, 3))
                self.encoder = self.build_encoder((self.max_seq_length, 3))
                self.decoder = self.build_decoder(self.max_seq_length)

            self.encoder.load_weights(filepath + '_encoder')
            self.decoder.load_weights(filepath + '_decoder')
        except Exception as e:
            raise Exception(f"Could not load model weights: {str(e)}")