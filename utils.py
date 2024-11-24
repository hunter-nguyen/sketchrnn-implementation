import matplotlib.pyplot as plt
import numpy as np

def plot_stroke(stroke_data, title="Stroke Sequence"):
    """Visualize a single stroke sequence"""
    # Convert deltas back to absolute coordinates
    coords = np.cumsum(stroke_data[:, :2], axis=0)

    plt.figure(figsize=(10, 10))
    plt.title(title)

    # Plot each stroke segment
    curr_stroke = []
    for i, (x, y, pen) in enumerate(zip(coords[:, 0], coords[:, 1], stroke_data[:, 2])):
        curr_stroke.append((x, y))

        if pen == 1 or pen == 2 or i == len(coords) - 1:
            # Draw the current stroke
            points = np.array(curr_stroke)
            plt.plot(points[:, 0], points[:, 1], 'b-')
            curr_stroke = []

    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_stroke_simplified(stroke_data, title=""):
    """Vector representation with clear pixel coordinates"""
    if len(stroke_data) == 0:
        return plt

    # Convert deltas to absolute canvas coordinates
    coords = np.zeros((len(stroke_data), 2))
    coords[0] = stroke_data[0, :2]
    coords[1:] = np.cumsum(stroke_data[1:, :2], axis=0)

    plt.figure(figsize=(12, 5))

    # Original Drawing (Left subplot)
    plt.subplot(1, 2, 1)
    plt.title("Original Drawing\n(Canvas Coordinates)")

    # Plot the actual drawing path
    current_stroke = []
    for i, (x, y, pen) in enumerate(zip(coords[:, 0], coords[:, 1], stroke_data[:, 2])):
        if pen == 1 or i == 0:  # Start new stroke
            if current_stroke:
                points = np.array(current_stroke)
                plt.plot(points[:, 0], points[:, 1], 'k-', linewidth=2)
            current_stroke = [(x, y)]
        else:
            current_stroke.append((x, y))

    if current_stroke:
        points = np.array(current_stroke)
        plt.plot(points[:, 0], points[:, 1], 'k-', linewidth=2)

    plt.grid(True)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")

    # Movement Vectors (Right subplot)
    plt.subplot(1, 2, 2)
    plt.title("Movement Vectors\n(Pixel Movements)")

    # Plot each movement vector
    for i in range(len(stroke_data)-1):
        dx, dy, pen = stroke_data[i]
        x, y = coords[i]

        # Draw movement vector
        color = 'blue' if pen == 0 else 'red'
        plt.arrow(x, y, dx, dy,
                 head_width=2,
                 head_length=3,
                 fc=color,
                 ec=color,
                 alpha=0.6,
                 length_includes_head=True)

        # Annotate first few movements with pixel distances
        if i < 5:
            plt.annotate(f'Move {i+1}:\nΔx={dx:.0f}px\nΔy={dy:.0f}px',
                        (x, y),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.grid(True)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")

    # Add legend
    plt.plot([], [], 'blue', label='Drawing (Pen Down)')
    plt.plot([], [], 'red', label='Moving (Pen Up)')
    plt.legend()

    plt.tight_layout()
    return plt

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