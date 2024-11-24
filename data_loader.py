import json
import numpy as np
import requests
import tensorflow as tf

def load_ndjson_data(file_path, max_samples=1000):
    strokes_data = []

    try:
        # Handle URL or local file
        if file_path.startswith('http'):
            response = requests.get(file_path)
            lines = response.text.split('\n')
        else:
            with open(file_path, 'r') as f:
                lines = f.readlines()

        for i, line in enumerate(lines):
            if i >= max_samples or not line.strip():
                break
            try:
                sketch = json.loads(line)
                stroke_data = sketch['drawing']
                formatted_stroke = format_stroke(stroke_data)
                strokes_data.append(formatted_stroke)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line {i}: {e}")
                continue

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    return strokes_data

def format_stroke(stroke_data):
    """Convert Quick Draw stroke format to SketchRNN format"""
    result = []
    for stroke in stroke_data:
        # Each stroke is [x_coords, y_coords, pen_state]
        x_coords = stroke[0]  # List of x coordinates
        y_coords = stroke[1]  # List of y coordinates

        for i in range(len(x_coords)):
            if i == 0 and len(result) > 0:
                # Pen up for new stroke, except for the first point
                result.append([x_coords[i], y_coords[i], 1])
            else:
                # Pen down while drawing
                result.append([x_coords[i], y_coords[i], 0])

    # Add end of drawing token
    result.append([0, 0, 2])

    # Convert to numpy array
    result = np.array(result, dtype=np.float32)

    # Convert to deltas (differences between consecutive points)
    result[1:, 0:2] -= result[:-1, 0:2]

    return result

def normalize_strokes(strokes_data):
    """Normalize stroke data to have zero mean and unit variance"""
    # Only normalize the x and y coordinates, not the pen state
    coords = strokes_data[:, :2]
    mean = np.mean(coords, axis=0)
    std = np.std(coords, axis=0)
    # Avoid division by zero
    std = np.maximum(std, 1e-6)

    normalized_strokes = strokes_data.copy()
    normalized_strokes[:, :2] = (coords - mean) / std
    return normalized_strokes

def validate_strokes(strokes_data):
    """Check if the stroke data is valid"""
    if len(strokes_data) == 0:
        return False

    # Check if pen states are valid (0, 1, or 2)
    pen_states = strokes_data[:, 2]
    if not np.all(np.isin(pen_states, [0, 1, 2])):
        return False

    # Check if coordinates are finite
    if not np.all(np.isfinite(strokes_data[:, :2])):
        return False

    return True

def process_data(file_path, max_samples=1000, max_seq_length=None):
    """Complete data processing pipeline"""
    raw_data = load_ndjson_data(file_path, max_samples)

    if raw_data is None:
        raise ValueError("Failed to load data")

    # Find max sequence length if not provided
    if max_seq_length is None:
        max_seq_length = max(len(stroke) for stroke in raw_data)

    # Pad sequences to same length using TensorFlow
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(
        raw_data,
        maxlen=max_seq_length,
        padding='post',
        dtype='float32',
        value=0.0
    )

    return padded_data, max_seq_length

def convert_to_strokes(image_data):
    """Convert canvas image data to stroke format"""
    if image_data is None:
        return np.array([])

    # Get the alpha channel to detect drawn pixels
    alpha = image_data[:, :, 3]

    # Find contiguous drawing segments
    strokes = []
    current_stroke = []

    # Scan the image row by row
    for y in range(alpha.shape[0]):
        for x in range(alpha.shape[1]):
            if alpha[y, x] > 0:  # If pixel is drawn
                if not current_stroke:  # Start new stroke
                    current_stroke = [(x, y)]
                else:
                    # Check if point is connected to previous
                    prev_x, prev_y = current_stroke[-1]
                    if abs(x - prev_x) <= 1 and abs(y - prev_y) <= 1:
                        current_stroke.append((x, y))
                    else:
                        # End current stroke and start new one
                        if len(current_stroke) > 1:
                            strokes.append(np.array(current_stroke))
                        current_stroke = [(x, y)]

    # Add last stroke if exists
    if current_stroke and len(current_stroke) > 1:
        strokes.append(np.array(current_stroke))

    # Convert to SketchRNN format [dx, dy, pen_state]
    result = []
    for stroke in strokes:
        # Convert absolute coordinates to deltas
        stroke = np.array(stroke)
        delta = np.zeros((len(stroke), 3))
        delta[1:, :2] = stroke[1:] - stroke[:-1]  # Calculate differences
        delta[0, :2] = stroke[0] - (stroke[0] if len(result) == 0 else result[-1][:2])

        # Add pen states
        delta[:, 2] = 0  # pen down
        if len(result) > 0:
            delta[0, 2] = 1  # pen up for new stroke

        result.extend(delta.tolist())

    # Add end token
    if result:
        result.append([0, 0, 2])

    return np.array(result, dtype=np.float32)