import cv2
import numpy as np
import streamlit as st
import imutils
import av

from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from tensorflow.keras.models import load_model

from setting import Caffemodel, PROTOTXT, SACMaskNet

net = cv2.dnn.readNetFromCaffe(PROTOTXT,Caffemodel)

#Funciones

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    )


cnn = load_model(SACMaskNet)

def app_object_detection():

    class OverwritePrediction(VideoProcessorBase):
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
          img = frame.to_ndarray(format="bgr24")

          img = imutils.resize(img, width=300)
          (h, w) = img.shape[:2]

          blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
          net.setInput(blob)
          detections = net.forward()

          for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                    
              box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
              (startX, startY, endX, endY) = box.astype("int")


              crop_img = img[startY:endY, startX:endX]
              data = []
              predict = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
              resized = cv2.resize(predict, (100, 100))
              data.append(resized)
              data = np.array(data)/255
              valor = cnn.predict_classes(data)
              if valor == 0:
                text = "Bad Mask"
                color = (0, 186, 251)
              elif valor == 1:
                text = 'With Mask'
                color = (70, 152, 0)
              elif valor == 2:
                text = 'Without Mask'
                color = (0,0,255)

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
          return av.VideoFrame.from_ndarray(img, format="bgr24")



    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=OverwritePrediction,
        async_processing=True,
    )


#App

#ASÍ QUEDA:
st.title('Face Mask Detection System Applying Convolutional Neural Networks')
#st.title('Sistema de Detección de Mascarillas Faciales aplicando Redes Neuronales Convolucionales')
st.write("")
#st.write("Esta aplicación identifica en tiempo real si tiene o no mascarilla.")
st.write("This application identifying in real-time if people use correctly/incorrectly their masks")
st.write("")
#st.write("Estimado usuario, para hacer uso de esta aplicación debe dar hacer click en START,   luego debe permitir el acceso a la cámara de su dispositivo, hecho esto, espere a que conecte con el servidor de streamlit.")
st.write("Dear user, to use this app you must click START, after this allow the access to your camera of your device, finally wait until the camera connects to streamlit server.")
object_detection_page = "Real Time Mask Detection"

app_mode = object_detection_page

st.subheader(app_mode)
if app_mode == object_detection_page:
        app_object_detection()
        st.write("")
       # st.write("El funcionamiento de la cámara web está basado en https://github.com/whitphx/streamlit-webrtc")
        st.write("Camera operation is based on https://github.com/whitphx/streamlit-webrtc")       

