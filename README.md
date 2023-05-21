# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
 
2. Upload the dataset in the compiler and read the dataset.

3. Find head,info and null elements in the dataset.
 
4. Using LabelEncoder and DecisionTreeRegressor , find MSE and R2 of the dataset.
 
5. Predict the values and end the program.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Santhosh U
RegisterNumber:  212222240092
*/

import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:
### 1. data.head()
![Output1](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477975/0a8fa74d-66a9-432e-99cc-7cf6c8ea5d6e)

### 2. data.info()
![Output2](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477975/71043c96-2d04-4be8-b894-401592a4fa98)

### 3. isnull() and sum()
![Output3](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477975/e91c0365-d421-42cb-84a3-ed0e4a5118e1)

### 4. data.head() for salary 
![Output4](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477975/6c11d1bc-dad8-4ad2-805b-8ccc61c9d2d2)

### 5. MSE value
![Output5](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477975/86cbcbfe-d8fe-4d47-be0c-84b8b45a2aec)

### 6. r2 value
![Output6](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477975/93d94727-de0d-4ed9-bad2-5c0792da65d8)

### 7. data prediction
![Output7](https://github.com/SanthoshUthiraKumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119477975/bf84c2c8-c4d6-4136-8835-4349476abc01)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
