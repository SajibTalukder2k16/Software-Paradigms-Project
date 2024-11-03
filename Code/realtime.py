# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:17:24 2024

@author: sajib
"""

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from inference_sdk import InferenceHTTPClient

# Initialize API client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="HXblEa6WLoZkYmTZA7wI"
)

def get_prediction(image_path, model_id="canned-food-surface-defect/1"):
    try:
        # Make an API call to get the prediction
        result = CLIENT.infer(image_path, model_id=model_id)
        if "predictions" not in result or not result['predictions']:
            print("No predictions found.")
            return None
        return result['predictions'][0]
    except Exception as e:
        print(f"Error fetching prediction: {e}")
        return None

def draw_bounding_box(image, prediction):
    # Calculate bounding box coordinates
    x, y = int(prediction['x']), int(prediction['y'])
    width, height = int(prediction['width']), int(prediction['height'])
    start_point = (x - width // 2, y - height // 2)
    end_point = (x + width // 2, y + height // 2)
    
    # Draw bounding box and label
    color = (255, 0, 0)  # Red color for the box
    cv2.rectangle(image, start_point, end_point, color, 2)
    label = f"{prediction['class']} ({prediction['confidence']:.2f})"
    cv2.putText(image, label, (start_point[0], start_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def process_frame(frame):
    # Save the current frame as a temporary image file
    temp_image_path = "temp_frame.png"
    cv2.imwrite(temp_image_path, frame)

    # Get prediction for the current frame
    prediction = get_prediction(temp_image_path)
    if prediction:
        frame = draw_bounding_box(frame, prediction)
    
    return frame

def start_camera():
    # Initialize Tkinter window
    window = tk.Tk()
    window.title("Real-Time Defect Detection")
    
    # Set up video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Tkinter label for displaying the video feed
    label = tk.Label(window)
    label.pack()

    def update_frame():
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            window.after(10, update_frame)
            return
        
        # Process the frame (get prediction and draw bounding box)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = process_frame(frame)

        # Convert the frame to PIL format and then to Tkinter-compatible format
        image = Image.fromarray(frame)
        image_tk = ImageTk.PhotoImage(image, master=window)

        # Update the label with the new image
        label.config(image=image_tk)
        label.image = image_tk  # Keep a reference to avoid garbage collection

        # Schedule the next frame update
        window.after(10, update_frame)

    # Start updating frames
    update_frame()

    # Run the Tkinter event loop
    window.mainloop()

    # Release the camera when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()
