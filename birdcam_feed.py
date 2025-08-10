import cv2
import logging
from ultralytics import YOLO
import time
import datetime
import os
import torch

# --- Global variables for logging frequency ---
LAST_LOG_TIME = {}  # Stores the last time a specific object was logged
LOG_INTERVAL_SECONDS = 1 # Log a bird detection once every 30 seconds
LOG_DIR = 'bird_detection_logs'
TARGET_CLASS_NAMES = ['bird', 'mouse', 'cat', 'dog']

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}") # 0 for the first GPU

def create_log_directory(log_dir):
    """Creates the specified log directory if it doesn't exist."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")


def display_rtsp_feed(rtsp_url):
    """
    Reads an RTSP camera feed and displays it using OpenCV.

    Args:
        rtsp_url (str): The RTSP URL of the camera feed.
                        Example: 'rtsp://username:password@ip_address:port/path'
    """
    print(f"Attempting to open RTSP stream from: {rtsp_url}")

    # Create a VideoCapture object
    cap = cv2.VideoCapture(rtsp_url)
    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video stream from {rtsp_url}")
        print("Please check the RTSP URL, network connectivity, and camera status.")
        return
    print("RTSP stream opened successfully. Initializing YOLOv8 Model...")

    model=YOLO('yolov8s.pt')
    # Explicitly move the model to the GPU if CUDA is available
    if torch.cuda.is_available():
        model.to('cuda')
        print("Model moved to GPU (CUDA).")
    else:
        print("CUDA not available. Model will run on CPU.")

    # Get the class names from the model
    class_names = model.names

    # This dictionary will store the mapping from your desired class name (string)
    # to its corresponding integer class ID from the YOLO model.
    mapped_target_class_ids = {}

    for desired_name in TARGET_CLASS_NAMES:
        found_id = None
        for idx, model_name in class_names.items():
            if model_name.lower() == desired_name.lower():
                found_id = idx
                break
        
        if found_id is not None:
            mapped_target_class_ids[desired_name.lower()] = found_id
            print(f"YOLOv8 model will look for: '{desired_name}' (ID: {found_id})")
        else:
            print(f"Warning: '{desired_name}' not found in the YOLOv8 model's class list. It will not be detected.")

    if not mapped_target_class_ids:
        print("No detectable classes from TARGET_CLASS_NAMES found in the YOLO model. Exiting.")
        return
            
    print("Press 'q' to quit.")

    # Create the log directory
    create_log_directory(LOG_DIR)

    # Variables to track presence for logging frequency (per detected class)
    # Initialize for only the classes that were actually found in the model's names
    object_present_since = {obj_name: None for obj_name in mapped_target_class_ids.keys()}

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Perform inference on the frame
        results = model(frame, verbose=False)

        detected_in_current_frame = {obj_name: False for obj_name in mapped_target_class_ids.keys()}
        
        annotated_frame = frame.copy() # Create a copy for drawing bounding boxes

        # A simple way to assign different colors to different classes for visualization
        # You can expand this for more distinct colors if you have many classes
        colors = {
            'bird': (0, 255, 0),    # Green
            'mouse': (255, 0, 0), # Blue
            'cat': (0, 255, 255),   # Yellow
            'dog': (255, 255, 0)    # Cyan
            # Add more colors for other classes as needed
        }

        for r in results:
            boxes = r.boxes # Bounding boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Iterate through our *mapped* target class IDs
                for desired_name_lower, desired_id in mapped_target_class_ids.items():
                    if cls_id == desired_id and confidence > 0.5:  # Adjust confidence threshold as needed
                        detected_in_current_frame[desired_name_lower] = True
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Determine color for the bounding box
                        draw_color = colors.get(desired_name_lower, (0, 255, 255)) # Default to yellow if no specific color defined

                        # Draw bounding box and label on the frame
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), draw_color, 2)
                        label = f"{desired_name_lower.capitalize()} {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
                        break # Found a match for this box, move to the next box

        current_time = datetime.datetime.now()
        for obj_name_lower in mapped_target_class_ids.keys():
            should_log_and_save = False

            if detected_in_current_frame[obj_name_lower]:
                if object_present_since[obj_name_lower] is None:
                    # Object just appeared, log and save immediately
                    object_present_since[obj_name_lower] = current_time
                    should_log_and_save = True
                else:
                    # Object is still present, check if it's time to log again
                    if obj_name_lower not in LAST_LOG_TIME or \
                       (current_time - LAST_LOG_TIME[obj_name_lower]).total_seconds() >= LOG_INTERVAL_SECONDS:
                        should_log_and_save = True
            else:
                # Object is not detected in the current frame, reset its presence tracker
                object_present_since[obj_name_lower] = None

            if should_log_and_save:
                timestamp_str = current_time.strftime('%Y%m%d_%H%M%S')
                log_message = f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] {obj_name_lower.capitalize()} detected! Saving image..."
                print(log_message)
                
                # Construct filename: {obj_name}_YYYYMMDD_HHMMSS.jpg
                image_filename = os.path.join(LOG_DIR, f"{obj_name_lower}_{timestamp_str}.jpg")
                
                # Save the annotated frame
                cv2.imwrite(image_filename, annotated_frame)
                print(f"Saved image: {image_filename}")
                
                LAST_LOG_TIME[obj_name_lower] = current_time # Update last log time for this specific object


        # Display the resulting frame (with or without annotations)
        cv2.imshow('RTSP Camera Feed with Bird Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rtsp_camera_url = 'rtsp://user:password@192.168.12.167:554/stream1' # Example Placeholder

    display_rtsp_feed(rtsp_camera_url)