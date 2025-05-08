# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.
## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MOHAMED RASHITH S
RegisterNumber: 212223243003
*/
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data
data.shape
x=data['v2'].values
y=data['v1'].values
x.shape
y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
## Data
![image](https://github.com/user-attachments/assets/26390abb-a3c1-43e0-ab45-d7d7876c1b11)

## Shape()
![image](https://github.com/user-attachments/assets/99e9a0e8-744d-42b2-89d6-a1e6471732b0)

## x.shape()
![image](https://github.com/user-attachments/assets/14f59666-0cad-4965-96b4-e932dfbe1bd5)

## y.shape()
![image](https://github.com/user-attachments/assets/5e2e5373-eee7-4f3c-a884-f4a9b082f75c)

## x_train and x_train.shape()
![image](https://github.com/user-attachments/assets/2b8c9b79-cf19-40ab-9034-563b70e82dd1)
![image](https://github.com/user-attachments/assets/8c804797-b87f-444d-917b-21ce0254879b)

## y_pred value
![image](https://github.com/user-attachments/assets/819ee38c-676a-4dc3-a938-df3313b5eb80)

## Accuracy
![image](https://github.com/user-attachments/assets/0db8c237-a3f8-4e72-91fd-76e6d9b21d82)

## Confusion matrix
![image](https://github.com/user-attachments/assets/0df9c09d-9100-4908-ae27-2e9bd2b77a48)

## Classification Report 
![image](https://github.com/user-attachments/assets/970f1d7a-23d1-4364-9cb7-4fddb4520f9f)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
