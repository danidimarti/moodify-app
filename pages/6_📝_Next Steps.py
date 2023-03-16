import streamlit as st
from PIL import Image
import base64

with open('style.css') as f: 
    st.markdown(f'<style>{f.read()} </style>', unsafe_allow_html=True)

st.markdown( """ <style> .sidebar .sidebar-content {color: black} </style> """, unsafe_allow_html=True, )

container_style = """
    ..sidebar .sidebar-content {
        background-color: #212121;
        border: 1px solid #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        color: #D3D3D3;
        font-size: 1.1rem;
        font-family: 'Open Sans', sans-serif;
        
    } """

st.markdown("<h1 style='color:#9FE1B4; font-family: serif;'> Next Steps</h1>", unsafe_allow_html=True)
file_ = open("imgs/Next Steps.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="model-gif" width="486.5" height="192.5"><br> <br>',
    unsafe_allow_html=True
)
st.write(
    "<div style='font-size:1.2rem'>"
    "Moodify is an application that leverages facial recognition technology to personalize music recommendations on Spotify. <br><br> By analysing the user\'s facial expressions in real-time using the device\'s camera, Moodify can infer the user\'s current mood and suggest songs that match their emotional state. This project explores the opportunity we have to harness the power of computer vision to create more meaningful and personalized interactions with technology.\n My aim was to tackle a more the human aspect of data analytics by going beyond the quantitative analysis of data and incorporating the qualitative aspects of human experience, such as perception, cognition and emotion. <br><br> <b> As we incorporate aspects of human comprehension into data analytics we can gain deeper insights into complex phenomena such as social interactions, cultural norms and psychological states, that are not easily captured by numerical data alone.</b>\
    <div style='text-align: center; font-size: 1.5rem; font-weight: bold;'><br><br>But how can we teach a computer to recognize emotions?</div>\
    </div>",
    unsafe_allow_html=True
)