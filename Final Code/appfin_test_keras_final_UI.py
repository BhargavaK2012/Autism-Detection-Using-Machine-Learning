import streamlit as st
import numpy as np
import pickle
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import tensorflow
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from keras.models import load_model
from keras.preprocessing import image as keras_image

# Load the machine learning model
try:
   model = load_model(r'my_model.keras')
except IndexError:
    print("Reload the page")
    model = load_model(r'my_model.keras')

# Resize the image
def resizee(width, height):
    target_size = (128, 128)
    target_ratio = target_size[0] / target_size[1]
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width / target_ratio)

    # Resize to 128x128
    if new_width > target_size[0]:
        new_width = target_size[0]
        new_height = int(new_width / current_ratio)
    elif new_height > target_size[1]:
        new_height = target_size[1]
        new_width = int(new_height * current_ratio)

    return new_width, new_height

# Function to predict autism
def predict_autism(image):
    img = keras_image.img_to_array(image)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    #st.write(model)
    #st.write(prediction)
    if prediction[0][0] > 0.5:
        prediction= 'Autistic'
    else:
        prediction= 'Non-autistic'
    return prediction

def main():

    # Streamlit UI
    st.title('Autism Detection')
    st.write("This application detects autism from uploaded images or captured live pictures.")
    st.write("Please select one of the options below:")

    # Option to upload image or capture from webcam
    option = st.radio("Select Option:", ("Upload Image", "Capture Live Picture"))

    if option == "Upload Image":
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:

            filename = uploaded_file.name
            img = Image.open(uploaded_file)
            # st.write(img)
            st.image(img, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            width, height = img.size
            st.write(f"Original Image Size: {width}x{height}")

            size = (128, 128)
            image = img.resize(size)
            
            st.write("Classifying...")
            prediction = predict_autism(image)
            st.write(f'The person in the image is predicted to be: {prediction}')

            #classes = ["Autistic", "Non-Autistic"]
            #prediction = model.predict(image_array)
            #predict_class = np.argmax(prediction)
            #st.write("Classifying...")
            #st.write(f'The person in the image is predicted to be: {prediction}')

            
            #classes=["Autistic","Non-Autistic"]
            #prediction=model.predict(np.array([image]))
            #predict_class=np.argmax(prediction)
            #st.write("Classifying...")
            #prediction = predict_autism(image)
            #st.write(f'The person in the image is predicted to be: {classes[predict_class]}')






    if option == "Capture Live Picture":
        st.subheader("Capture Live Picture")
        st.write("Please press the button below to capture the live picture.")

        # Open a video capture object for webcam
        cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

        if not cap.isOpened():
            st.error("Error: Unable to open webcam.")
        
        # Initialize the captured image
        captured_image = None
        
        # Display the frame in the Streamlit app
        frame_placeholder = st.empty()

        # Initialize the capture button outside the loop
        capture_button = st.button("Capture")

        # Read frames from the webcam in a loop
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()

            # Check if the frame is read successfully
            if not ret:
                st.error("Error: Unable to read frame from webcam.")
                break
            
            # Display the live webcam feed
            frame_placeholder.image(frame, channels="BGR", caption="Live Webcam Feed")

            # Check if the capture button is pressed
            if capture_button:
                captured_image_click = frame
                st.write(captured_image_click.shape)
                captured_image=cv2.resize(captured_image_click,dsize=(128,128))
                st.write(captured_image.shape)
                break

        if captured_image is not None:
            st.image(captured_image, channels="BGR", caption="Captured Image")
            st.write("Classifying...")
            prediction = predict_autism(captured_image)
            st.write(f'The person in the image is predicted to be: {prediction}')
        else:
            st.warning("No image captured.")

        cap.release()

        # Add a refresh page button
        if st.button("Refresh Page"):
            # Run JavaScript to refresh the page
            st.write(
                "<script type='text/javascript'>"
                "window.location.reload();"
                "</script>",
                unsafe_allow_html=True,
            )





if __name__ == "__main__":
    main()

link_text = "Made with ❤️ by Ardra,  Ramya Shree, Avaneesh S, and Bhargava K"
link = f'<div style="display: block; text-align: center; margin-top:30%; padding: 5px;">{link_text}</div>'
st.write(link, unsafe_allow_html=True)