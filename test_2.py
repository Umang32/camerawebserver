import cv2
import numpy as np
import urllib.request
import time

# Configuration
ESP32_CAM_URL = ''
YOLO_WH_T = 320  # Input size for YOLO network
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Initialize YOLO network
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load COCO class names
with open('coco.names', 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')


def process_frame(img):
    # Create blob from image and process through network
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (YOLO_WH_T, YOLO_WH_T),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Forward pass through network
    outputs = net.forward(output_layers)
    return outputs


def draw_detections(outputs, img):
    h, w = img.shape[:2]
    boxes = []
    confidences = []
    class_ids = []

    # Process each output layer
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESHOLD:
                # Scale bounding box coordinates to image size
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")

                # Calculate top-left corner coordinates
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences,
                               CONF_THRESHOLD, NMS_THRESHOLD)

    # Draw detections if any exist
    if len(indices) > 0:
        for i in indices.flatten():  # Convert to 1D array
            x, y, w, h = boxes[i]

            # Draw bounding box and label
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(img, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# Main processing loop
def main():
    cv2.namedWindow('ESP32-CAM Detection', cv2.WINDOW_NORMAL)

    while True:
        try:
            # Fetch frame from ESP32-CAM
            with urllib.request.urlopen(ESP32_CAM_URL, timeout=2) as response:
                img_array = np.array(bytearray(response.read()), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Received empty frame")
                    continue

                # Process frame through YOLO
                outputs = process_frame(frame)
                draw_detections(outputs, frame)

                # Display result
                cv2.imshow('ESP32-CAM Detection', frame)

                # Exit on 'q' key
                if cv2.waitKey(1) == ord('q'):
                    break

        except urllib.error.URLError as e:
            print(f"Connection error: {e.reason}")
            print("Retrying in 3 seconds...")
            time.sleep(3)

        except Exception as e:
            print(f"Critical error: {str(e)}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()