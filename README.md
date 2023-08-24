# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python.
2. Set variables for assigning data set values.
3. Import Linear Regression from the sklearn.
4. Assign the points for representing the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtain the LinearRegression for the given data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mathiyazhagan.A
RegisterNumber:  212222240063
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('student_scores.csv')


data.head()
print("Data Head :\n" ,data.head())
data.tail()
print("\nData Tail :\n" ,data.tail())


x=data.iloc[:,:-1].values  
y=data.iloc[:,1].values

print("\nArray value of X:\n" ,x)
print("\nArray value of Y:\n", y)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0 )

regressor=LinearRegression() 
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test) 

print("\nValues of Y prediction :\n",y_pred)

print("\nArray values of Y test:\n",y_test)


print("\nTraining Set Graph:\n")
plt.scatter(x_train,y_train,color='red') 
plt.plot(x_train,regressor.predict(x_train),color='green') 
plt.title("Hours Vs Score(Training set)") 
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

y_pred=regressor.predict(x_test) 

print("\nTest Set Graph:\n")
plt.scatter(x_test,y_test,color='red') 
plt.plot(x_test,regressor.predict(x_test),color='green') 
plt.title("Hours Vs Score(Test set)") 
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

import sklearn.metrics as metrics

mae = metrics.mean_absolute_error(x, y)
mse = metrics.mean_squared_error(x, y)
rmse = np.sqrt(mse)  

print("\n\nValues of MSE, MAE and RMSE : \n")
print("MAE:",mae)
print("MSE:", mse)
print("RMSE:", rmse)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

df.head():


![Output1](https://github.com/Mathiofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787327/1a0c8cbd-643c-4eae-9fe0-a770309e9ed7)
df.tail():

![Output2](https://github.com/Mathiofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787327/c3065a4c-02ee-45c7-bcf3-f319dc16b314)

Array value of X:

![Screenshot 2023-08-24 091934](https://github.com/Mathiofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787327/bf262ace-7e76-4745-b78b-4d2e3eface8a)

Array value of Y:

![image](https://github.com/Mathiofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787327/4d76f0ec-b974-4146-a41c-ccec299707cc)

Values of Y prediction:

![Output5](https://github.com/Mathiofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787327/27428d72-a684-4741-aa5c-c2cf23e19bb9)

Array values of Y test:

![Output6](https://github.com/Mathiofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787327/f528e242-6281-455d-b8a6-2e936c909c63)


Training Set Graph:

![Output7](https://github.com/Mathiofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787327/5e5ec5aa-4176-4da0-aff2-710a2abc62b1)

Values of MSE, MAE and RMSE:

![Output9](https://github.com/Mathiofficial/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118787327/858344e7-1340-4141-8ca4-b59ac26d18d6)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
