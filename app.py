import gradio as gr
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("models/best.pt")


def detect(image):
    results = model(image)

    # Get annotated image
    output_image = results[0].plot()

    # Detection information
    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        detections.append(
            f"{class_name}: {confidence:.2f}"
        )

    if detections:
        text = "\n".join(detections)
    else:
        text = "No objects detected."

    return output_image, text


# Gradio interface
app = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Detected Objects")
    ],
    title="Person & Vehicle Detection",
    description="Upload an image to detect persons and vehicles using YOLO."
)

app.launch(share=True)