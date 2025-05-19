import cv2
from ultralytics import YOLO

# Path to the image file
image_path = r"img1.jpg"

# Load the YOLO model
model = YOLO("yolov8m.pt")

# Parameters

FOCAL_LENGTH = 360
KNOWN_WIDTH = 60

# Read class list from coco.txt
with open("coco.txt", "r") as file:
    class_list = file.read().split("\n")

# Detection colors for each class
detection_colors = [(0, 255, 0)] * len(class_list)


def perform_object_detection(image, confidence_threshold=0.8):
    # Perform object detection on the image
    results = model(image, show=True)

    # Filter detections based on confidence threshold
    if results and any(res.boxes for res in results):
        filtered_detections = []
        for res in results:
            for bbox in res.boxes:
                if bbox.conf >= confidence_threshold:
                    filtered_detections.append(res)
                    break

        # Select the detection with the largest area
        if filtered_detections:
            max_area = 0
            index = 0
            for i, res in enumerate(filtered_detections):
                for bbox in res.boxes:
                    area = bbox.xyxy[0][2] * bbox.xyxy[0][3]
                    if area > max_area:
                        max_area = area
                        index = i
            return filtered_detections[index]
    return None


def annotate_image(image, detection):
    # Annotate the image with detection results
    for bbox in detection.boxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        class_id = int(bbox.cls[0])
        confidence = round(float(bbox.conf[0]), 2)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), detection_colors[class_id], 2)

        # Label with class and confidence
        label = f"{class_list[class_id]}: {confidence}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Read the image from file
image = cv2.imread(image_path)

# Perform object detection on the image
detection = perform_object_detection(image)

# Annotate the image with detection results if any
if detection:
    annotate_image(image, detection)
    print("Object detected and annotated.")
else:
    print("No object detected.")

# Display the image with annotations
cv2.imshow("Object Detection and Annotation", image)
cv2.waitKey(0)  # Press any key to close the window
cv2.destroyAllWindows()