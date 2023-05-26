import asyncio
import os
import numpy as np
import PIL.Image as Image
from PIL import Image, ImageFilter
import pandas as pd
import streamlit as st
from PIL import Image
import torch
import shutil

asyncio.set_event_loop(asyncio.new_event_loop())


# Define the object detection function
def object_detection(image_path):
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'./best.pt')
    model.classes = [1, 2, 3]
    model.multi_label = True
    model.max_det = 2
    model.conf = 0.0
    model.iou = 0.0

    # Load the input image
    image = Image.open(image_path)

    # Perform object detection on the image
    results = model(image, size=640)
    results.save()

    # Extract the number of detections for each object class
    detections = results.xyxy[0][:, -1].cpu().numpy()
    class_names = ['Locks', 'MCB', 'wires']
    num_detections = {}
    for class_idx, class_name in enumerate(class_names):
        num_detections[class_name] = np.sum(detections == class_idx)

    return num_detections


# Define the Streamlit app
def app():
    # Set the app title
    st.set_page_config(page_title='Object Detection', page_icon=':camera:')
    # Add a sidebar with an upload file input
    st.sidebar.title('Upload Image')
    uploaded_file = st.sidebar.file_uploader('hello', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')
    # If an image is uploaded, display the image with detected objects
    if uploaded_file is not None:
        # Display the uploaded image
        st.title('Original Image')
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        width = np.array(image).shape[1]
        height = np.array(image).shape[0]

        st.title('Image Properties')
        img_props = {
            # 'Size': np.array(size),
            'Resolution': (str(width) + "x" + str(height)),
            'Mode': image.mode,
            'Shape': np.array(image).shape,
            'Pixel Value Range': (np.array(image).min(), np.array(image).max()),
            'Format': image.format,
        }
        img_props_df = pd.DataFrame.from_dict(img_props, orient='index', columns=['Value'])
        st.table(img_props_df)
        # Perform object detection on the uploaded image
        st.title('Detected Objects')
        object_detection(uploaded_file)
        object_counts = object_detection(uploaded_file)
        st.write('Following Objects have been detected:')
        num_detections = object_detection(uploaded_file)
        for class_name, num_objects in num_detections.items():
            if class_name == 'wires':
                st.write(f'Wires (sets) :  {num_objects} ')
            else:
                st.write(f'{class_name} : {num_objects} ')

        st.image(Image.open('./runs/detect/exp/image0.jpg'))
        shutil.rmtree('runs')


# Run the Streamlit app
if __name__ == '__main__':
    app()

