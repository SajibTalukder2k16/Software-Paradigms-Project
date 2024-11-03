import cv2
import tkinter as tk
from PIL import Image, ImageTk
from inference_sdk import InferenceHTTPClient
import os

# Initialize API client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="HXblEa6WLoZkYmTZA7wI"
)

def get_prediction(image_path, model_id="canned-food-surface-defect/1"):
    # Make an API call to get the prediction
    result = CLIENT.infer(image_path, model_id=model_id)
    if "predictions" not in result or not result['predictions']:
        print("No predictions found.")
        return None
    return result['predictions'][0]

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

def display_image(image):
    # Initialize Tkinter window before creating the image
    window = tk.Tk()
    window.title("Defect Detection")
    
    # Convert OpenCV image to PIL format for Tkinter compatibility
    image = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image, master=window)  # Specify the master to avoid the error
    
    # Display the image in the Tkinter window
    label = tk.Label(window, image=image_tk)
    label.image = image_tk  # Keep a reference to avoid garbage collection
    label.pack()
    
    # Start the Tkinter event loop
    window.mainloop()


def main():
    image_path = "../Images/can.png"
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prediction = get_prediction(image_path)
    if prediction:
        image = draw_bounding_box(image, prediction)
        display_image(image)

if __name__ == "__main__":
    main()
