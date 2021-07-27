import pandas as pd
import numpy as np
import pickle



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
from sklearn.metrics import confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# saving model
pickle.dump(logmodel,open("trained_model.sa","wb"))






