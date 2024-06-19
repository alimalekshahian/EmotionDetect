import os
from keras.models import load_model
from keras.optimizers import Adam
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

# Function to detect emotions in a single image and save the result
def detect_emotion_save(image_path, output_path, model_path):
    """
    Detects emotions in a single image, saves the result with labels.

    Args:
        image_path: Path to the input image.
        output_path: Path to save the output image with emotion labels.
        model_path: Path to the pre-trained model.
    """
    # Load input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade classifier for face detection
    haar_cascade_path = os.path.join('../data/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in detected_faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64))  # Adjusted to the correct size
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Load the model
        model = load_model(model_path, compile=False)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Make a prediction on the ROI, then lookup the class
        predictions = model.predict(roi)[0]
        max_index = np.argmax(predictions)
        emotion_label = emotion_labels[max_index]

        # Draw the label and bounding box on the frame
        label_position = (x, y - 10)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the output image with marked emotion labels
    cv2.imwrite(output_path, image)
    print(f"Output image saved as {output_path}")

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Example usage:
if __name__ == '__main__':
    input_image_path = '../images/img1.jpg'  # Replace with your image path
    output_image_path = '../images/output_image_with_emotion_labels_img1.jpg'  # Replace with desired output path
    model_path = '../models/fer2013_mini_XCEPTION.102-0.66.hdf5'  # Replace with your model path

    detect_emotion_save(input_image_path, output_image_path, model_path)
