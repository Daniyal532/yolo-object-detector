import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("models/best.pt")

st.set_page_config(
    page_title="Person & Vehicle Detection",
    page_icon="🔍"
)

st.title("🔍 Person & Vehicle Detection")
st.write("Upload an image to detect persons and vehicles using YOLO.")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Detect Objects"):
        with st.spinner("Detecting objects..."):

            results = model(image)

            output_image = results[0].plot()

            st.subheader("Detection Result")
            st.image(
                output_image,
                caption="Detected Objects",
                use_container_width=True
            )

            st.subheader("Detected Objects")

            detections = []

            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                detections.append(
                    f"{class_name}: {confidence:.2f}"
                )

            if detections:
                for detection in detections:
                    st.write("✅", detection)
            else:
                st.write("No objects detected.")