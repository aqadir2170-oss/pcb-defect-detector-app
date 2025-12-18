import streamlit as st
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


# This changes the browser tab title and the favicon (icon)
st.set_page_config(
    page_title="PCB Inspection System",
 
)

# --- CONFIGURATION ---
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
MODEL_PATH = r'./ultralytics/runs/detect/train13/weights/best.onnx'

# Your specific class names
CLASS_NAMES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short_circuit", "Spur", "Spurious_Copper"]

# Define colors for different defects (RGB)
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 165, 0),  # Orange
    (255, 0, 255),  # Magenta
    (0, 255, 255)   # Cyan
]

@st.cache_resource
def load_model(path):
    return ort.InferenceSession(path, providers=['CPUExecutionProvider'])

def process_yolo_output(output, original_image_size, input_size=(640, 640)):
    output = np.squeeze(output) 
    output = output.transpose()  
    
    boxes, scores, class_ids = [], [], []
    img_width, img_height = original_image_size
    x_factor, y_factor = img_width / input_size[0], img_height / input_size[1]

    for row in output:
        classes_scores = row[4:]
        max_score = np.amax(classes_scores)
        
        if max_score >= CONF_THRESHOLD:
            class_id = np.argmax(classes_scores)
            x, y, w, h = row[0], row[1], row[2], row[3]
            
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            scores.append(float(max_score))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    
    final_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_detections.append({
                "box": boxes[i],
                "score": scores[i],
                "class_id": class_ids[i],
                "label": CLASS_NAMES[class_ids[i]] # Map ID to Name
            })
    return final_detections

# --- STREAMLIT UI ---
st.title("🔍 PCB Defect Detector")
st.write("Detecting: Missing_ho , Mouse_bite , Open_circuit, Short_circuit, Spur, Spurious_Copper")

uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    if st.button('Run Inspection'):
        session = load_model(MODEL_PATH)
        
        # Inference
        img_resized = image.resize((640, 640))
        input_data = np.array(img_resized).astype('float32') / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})

        # Post-process
        detections = process_yolo_output(outputs[0], image.size)

       # 4. Visualization
        draw = ImageDraw.Draw(image)
        
        # Adjust font size based on image size for better readability
        try:
            font_size = int(image.size[0] * 0.03)  # Dynamic font size (2% of image width)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        for det in detections:
            x, y, w, h = det['box']
            label_text = f"{det['label']} {det['score']:.2f}"
            color = COLORS[det['class_id'] % len(COLORS)]
            
            # 1. Draw Bounding Box (Thinner width for small PCB components)
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            
            # 2. Draw Label Background "Tab"
            # Get text dimensions using the modern textbbox method
            text_box = draw.textbbox((x, y), label_text, font=font)
            
            # Refine the label background (shrink it to fit the text tightly)
            # text_box is (left, top, right, bottom)
            label_rect = [text_box[0], text_box[1] - 5, text_box[2] + 10, text_box[3] + 2]
            
            # Ensure label doesn't go off-screen if detection is at the very top
            if y < 30:
                label_rect = [x, y, text_box[2] + 10, y + (text_box[3] - text_box[1]) + 5]
                text_pos = (x + 5, y + 2)
            else:
                text_pos = (x + 5, y - (text_box[3] - text_box[1]) - 8)
                label_rect = [x, text_pos[1] - 2, text_box[2] + 10, y]

            draw.rectangle(label_rect, fill=color)
            
            # 3. Draw Text
            draw.text(text_pos, label_text, fill="white", font=font)

        st.image(image, caption="Refined Inspection Result", use_container_width=True)