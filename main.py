
#import libarary
from click import File
import streamlit as st
from keras.models import load_model
from PIL import ImageOps, Image
from utli import classify,set_background

set_background("./bg/background.png")
#set_Title:
st.title("Pneunonia Classification")
#set_header:
st.header("Please upload an chest X-ray image")
#upload Classifer:
file=st.file_uploader("",type=['jpeg','jpg','png'])

#Load classifier:
model=load_model("./model/pneumonia_classifier.h5")
#Load class names:
with open("./model/labels.txt","r") as f:
    class_names=[a[:-1].split(" ")[1] for a in f.readlines()]
    f.close()

#display_image:

if file is not None:
    image=Image.open(file).convert("RGB")
    st.image(image,use_column_width=True)    


#Classifer_image:

    class_name, conf_score = classify(image, model, class_names)

#write_Classifer
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
