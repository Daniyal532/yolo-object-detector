from ultralytics import YOLO

# Load trained model
model = YOLO("models/best.pt")

# Run detection
results = model("images.jpeg")

# Show detected objects
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.2f}")

# Display result
results[0].show()