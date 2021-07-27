import pickle
import streamlit as st
import numpy as np
import pandas as pd


st.set_page_config(page_title="Survival Predicting Application",
layout="wide")
st.title("Survival Predicting")
st.subheader("Will you survive?")

p_class = st.selectbox("Class? ", ("1","2","3"),index=2)

p_age = st.slider('Age?', 0.4, 80.0, 29.0,step=1.0)

p_fare = st.slider('Money Spent?', 0, 512, 32)

p_gender = st.radio("Gender ", ("Female","Male"),index=1)

if p_gender == "Male":
    p_gender = 1
else:
    p_gender = 0


#Southampton
#Queenstown
#Cherbourg

# After leaving Southampton on 10 April 1912,
#  Titanic called at Cherbourg in France
#  and Queenstown (now Cobh) in Ireland,
#  before heading west to New York.

p_port = st.radio("Embarking Port", ("Southampton (England)","Queenstown (Ireland)","Cherbourg (France)"),index=0)
q = 0
s = 0
if p_port == "Southampton (England)":
    s = 1
elif p_port== "Queenstown (Ireland)":
    q = 1
    


p_alone = st.checkbox("Alone",value=True)



p_relatives = 0

if p_alone == False:
    p_alone = 0
    rels = st.text_input("Relatives? ",max_chars=2,value=1)
    if rels != None:
        p_relatives = int(rels)
    else:
        p_alone = 1
else:
    p_alone = 1


isSurv = st.button("OK")

if isSurv:

    logmodel = pickle.load(open("trained_model.sa","rb"))
    to_predict = pd.DataFrame(
    {"Pclass":int(p_class),
    "Age":np.float64(p_age),
    "Fare": np.float64(p_fare),
    "male":np.uint8(p_gender),
    "Q":np.uint8(q),
    "S":np.uint8(s),
    "relatives":int(p_relatives),
    "was_alone": p_alone,
    },index=[0])

    st.write(to_predict)

    res = logmodel.predict(to_predict)

    if res[0] == 0:
        st.write("Oh sorry!     you won't make it...")
    else:
        st.write("Congratulations!    you will make it!")



