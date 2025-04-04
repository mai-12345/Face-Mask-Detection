
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title(" Face Mask Detection Application")

model = load_model("face_mask_detection_model.h5")


uploaded_image=st.file_uploader(" Choose an Image....",type=['png', 'jpg', 'jpeg', 'webp'])

#define a function for face detectiozn and mask detection
def detect_and_predict_mask(image):
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces=face_cascade.detectMultiScale(image, scaleFactor= 1.1)

    prediction=[]

    for (x,y,w,h) in faces:
        face=image[y:y+h,x:x+w]
        face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face=cv2.resize(face,(128,128))
        face=np.array(face)/255.0
        face=np.expand_dims(face,axis=0)

        #predict mask/no mask
        prediction.append(model.predict(face))

    return faces,prediction

if uploaded_image is not None:
    file_bytes=np.asarray(bytearray(uploaded_image.read()),dtype = np.uint8)
    image=cv2.imdecode(file_bytes,1)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    faces,prediction=detect_and_predict_mask(image)

    for i ,(x,y,w,h) in enumerate(faces):
        (mask,withoutMask)=prediction[i][0]
        label='Mask' if mask>withoutMask else 'No Mask'

        color = (0,0,255) if label == 'Mask' else (255, 0, 0)
    
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(image, (x,y), (x+w, y+h), color, 1)

st.image(image, caption = 'Detected Image',use_container_width=True)










