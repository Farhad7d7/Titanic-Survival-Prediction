import re
import pandas as pd
import numpy as np
from pandas.core.dtypes.missing import notnull
import streamlit as st



train = pd.read_csv('titanic_train.csv')

# print(train.head())

def fill_age(dt):
    Age = dt[0]
    PClass = dt[1]
    
    if pd.isnull(Age):
        
        if PClass == 1:
            return 37
        elif PClass == 2:
            return 29
        elif PClass == 3:
            return 24
        
    else:
        return Age

def wasAlone(rels):

    isAlone = []
    for rel in rels:

        if rel > 0:
            isAlone.append(0)
        else:
            isAlone.append(1)

    return isAlone

train['Age'] = train[['Age','Pclass']].apply(fill_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
emb = pd.get_dummies(train['Embarked'],drop_first=True)
sx = pd.get_dummies(train['Sex'],drop_first=True)
train = pd.concat([train,sx,emb],axis=1)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.drop(['PassengerId'],axis=1,inplace=True)
pcalss = pd.get_dummies(train['Pclass'],drop_first=True)

train['relatives'] = train['SibSp'] + train['Parch']
train['was_alone'] = wasAlone(train['relatives'])
train.drop(['SibSp','Parch'],axis=1,inplace=True)
X = train.drop('Survived',axis=1)
y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report

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


isSurv = st.button("OK")

if isSurv:

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







