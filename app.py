import cv2
import numpy as np
import streamlit as st
import imutils
from streamlit import caching
import random

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from tensorflow.keras.models import load_model

from setting import Caffemodel, PROTOTXT, Maskmodel
#prototxt = 'deploy.prototxt'
#model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(PROTOTXT,Caffemodel)
#Funciones

###########################################################################################
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    )
############################################################################################




cnn = load_model(Maskmodel)


def app_object_detection():

    class OverwritePrediction(VideoProcessorBase):
        
        @st.cache(max_entries=10, ttl=3600)
        def transform(self, frame):
          img = frame.to_ndarray(format="bgr24")

          img = imutils.resize(img, width=300)
          (h, w) = img.shape[:2]

          blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
          net.setInput(blob)
          detections = net.forward()

          for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                    # extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
              box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
              (startX, startY, endX, endY) = box.astype("int")


##############
              crop_img = img[startY:endY, startX:endX]
              data = []
              predecir = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
              resized = cv2.resize(predecir, (100, 100))
              data.append(resized)
              data = np.array(data)/255
              valor = cnn.predict_classes(data)
              color =  (255,0,0)
              text = "patricio"
              if valor == 0:
                text = "Mal Puesta"
                text = "Bad Mask"
                color = (0, 186, 251)
              elif valor == 1:
                text = 'Mascarilla'
                text = 'With Mask'
                color = (70, 152, 0)
              elif valor == 2:
                text = 'Sin Mascarilla'
                text = 'Without Mask'
                color = (0,0,255)
#############
                    # display the prediction
                   # label = text: {round(confidence * 100, 2)}%"
              cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
              y = startY - 15 if startY - 15 > 15 else startY + 15
              cv2.putText(
                        img,
                        text,
                        (startX, y+10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )
          return img



    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=OverwritePrediction,
        async_processing=True,
    )

  


#App

#ASÍ QUEDA:
st.title(' Uso de algoritmos de Machine Learning para identificar el uso de mascarillas faciales')
st.write("")
st.write("Esta aplicación identifica en tiempo real si tiene o no mascarilla.")
st.write("")
st.write("Estimado usuario, para hacer uso de esta aplicación debe permitir el acceso a su cámara web.")
object_detection_page = "Real Time Mask Detection"

app_mode = object_detection_page

st.subheader(app_mode)
if app_mode == object_detection_page:
        app_object_detection()
