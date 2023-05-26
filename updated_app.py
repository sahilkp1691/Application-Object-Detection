#!/usr/bin/env python
# coding: utf-8

import cv2
import asyncio
import os
import openpyxl
import numpy as np
import PIL.Image as Image
from PIL import Image, ImageFilter
import torch
import shutil
import pandas as pd
import streamlit as st

asyncio.set_event_loop(asyncio.new_event_loop())

# Define the object detection function
def object_detection(image_path):
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'./best.pt')
    model.classes = [1,2,3]
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
    class_names = ['Security_seal','Circuit_breaker', 'Polarity']
    num_detections = {}
    for class_idx, class_name in enumerate(class_names):
        num_detections[class_name] = np.sum(detections == class_idx)

    return num_detections

def app():
    # Set the app title
    st.set_page_config(page_title='Object__ Detection', page_icon=':camera:')

    # Add a sidebar with an upload file input
    st.sidebar.title('Upload Image')
    uploaded_file = st.sidebar.file_uploader('hello', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')

    # If an image is uploaded, display the image with detected objects
    if uploaded_file is not None:
        # Display the uploaded image
        st.title('Original Image')
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        
        max_var = 1145.43
        # Blur
        gray = image.convert('L')
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        b_laplacian = gray.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=1))
        variance = np.var(np.array(b_laplacian))
        scaled_var = (variance / max_var) * 100
        max_value = 100  # Set the maximum possible blur value
        blur_scaled = int((scaled_var / max_value) * 100)
        st.write('Blur: ', blur_scaled)
        
        max_var = 4918.44
        # Sharpness
        s_laplacian = cv2.Laplacian(np.array(image), cv2.CV_64F)
        var = s_laplacian.var()
        s_scaled_var = (var / max_var) * 100
        max_value = 100  # Set the maximum possible sharpness value
        sharpness_scaled = int((s_scaled_var / max_value) * 100)
        st.write('Sharpness: ', sharpness_scaled)
        
        # Extract image properties
        width = np.array(image).shape[1]
        height = np.array(image).shape[0]
        img_props = {
            'Name': uploaded_file.name,
            'Resolution': f'{width}x{height}',
        }
        img_props_df = pd.DataFrame.from_dict(img_props, orient='index', columns=['Value'])
        
        # Perform object detection on the uploaded image
        object_counts = object_detection(uploaded_file)
        num_detections = {}
        for class_name, num_objects in object_counts.items():
            if class_name in ['Security_seal','Circuit_breaker', 'Polarity']:
                num_detections[class_name] = num_objects

        # Create DataFrame with image properties and object detections
        data = {
        'Name': img_props['Name'],
        'Resolution': img_props['Resolution'],
        'Polarity': 'Y' if 'Polarity' in object_counts and object_counts['Polarity'] > 0 else 'N',
        'Security_seal': 'Y' if 'Security_seal' in object_counts and object_counts['Security_seal'] > 0 else 'N',
        'Circuit_breaker': 'Y' if 'Circuit_breaker' in object_counts and object_counts['Circuit_breaker'] > 0 else 'N',
        'Photo Quality' : 'FAIL' if width<900 or height<1200 or blur_scaled > 50 or sharpness_scaled < 50 else 'PASS',
    }
        df = pd.DataFrame(data, index=[0])
        
                ## Write image properties and object detections to Excel file
        if os.path.exists('object_detections.xlsx'):
            # Read existing data into DataFrames
            with pd.ExcelFile('object_detections.xlsx') as reader:
                detections_df = pd.read_excel(reader, sheet_name='Object Detections')
                img_props_df = pd.read_excel(reader, sheet_name='Image Properties')
            # Append new data to existing DataFrames
            detections_df = pd.concat([detections_df, df], ignore_index=True)
            img_props_df = pd.concat([img_props_df, img_props_df.iloc[0:0].append(img_props_df.iloc[0], ignore_index=True)], ignore_index=True)
            # Write updated DataFrames to Excel file
            with pd.ExcelWriter('object_detections.xlsx') as writer:
                detections_df.to_excel(writer, sheet_name='Object Detections', index=False)
                img_props_df.to_excel(writer, sheet_name='Image Properties', index=False)
        else:
            with pd.ExcelWriter('object_detections.xlsx') as writer:
                df.to_excel(writer, sheet_name='Object Detections', index=False)
                img_props_df.to_excel(writer, sheet_name='Image Properties', index=False)
        
        # Display object detections
        st.title('Detected Objects')
        for class_name, num_objects in num_detections.items():
            if class_name == 'wires':
                st.write(f'Wires (sets): {num_objects}')
            else:
                st.write(f'{class_name}: {num_objects}')
        
        # Display processed image
        st.title('Processed Image')
        st.image(Image.open('./runs/detect/exp/image0.jpg'))
        shutil.rmtree('runs')


# Run the Streamlit app
if __name__ == '__main__':
    app()
