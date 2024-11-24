import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from utils import plot_stroke_simplified

def main():
    st.title("SketchRNN Demo")

    # Create canvas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Your Drawing")
        canvas_result = st_canvas(
            stroke_width=2,
            stroke_color='#000000',
            background_color='#ffffff',
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
        )

        # Real-time vector representation toggle
        show_vectors = st.checkbox('Show vector representation in real-time', value=False)

        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            if show_vectors or st.button('Show Vector Representation'):
                strokes = extract_strokes_from_json(canvas_result.json_data)
                if len(strokes) > 0:
                    with col2:
                        st.markdown("### Vector Representation")
                        plot_stroke_simplified(strokes, "Drawing as Vectors")
                        st.image(plot_to_image())

def extract_strokes_from_json(json_data):
    """Extract stroke data from canvas JSON data"""
    strokes = []
    current_stroke = []

    for obj in json_data["objects"]:
        if "path" in obj:
            points = obj["path"]
            for i, point in enumerate(points):
                if point[0] == "M":  # Move to (start of stroke)
                    if current_stroke:
                        strokes.append(np.array(current_stroke))
                    current_stroke = [(point[1], point[2])]
                elif point[0] == "L":  # Line to
                    current_stroke.append((point[1], point[2]))

    if current_stroke:
        strokes.append(np.array(current_stroke))

    # Convert to vector format [dx, dy, pen_state]
    result = []
    for stroke in strokes:
        # Convert absolute coordinates to deltas
        delta = np.zeros((len(stroke), 3))
        delta[1:, :2] = stroke[1:] - stroke[:-1]  # Calculate differences
        delta[0, :2] = stroke[0] - (stroke[0] if len(result) == 0 else result[-1][:2])

        # Add pen states
        delta[:, 2] = 0  # pen down
        if len(result) > 0:
            delta[0, 2] = 1  # pen up for new stroke

        result.extend(delta.tolist())

    if result:
        result.append([0, 0, 2])  # End token
        return np.array(result, dtype=np.float32)

    return np.array([])

def plot_to_image():
    """Convert matplotlib plot to image"""
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return buf

if __name__ == "__main__":
    main()