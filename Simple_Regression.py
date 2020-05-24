import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")
print(dataset)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values
print("<<<<<<<xxxxxxxxxxxxxxxx")
print(X)
print("<<<<<<<yyyyyyyyyyyyyyyy")
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y , test_size = 1/3 , random_state = 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
print(X_train)
print(Y_train)
regressor.fit(X_train, Y_train)
print(regressor.fit(X_train, Y_train))

y_pred = regressor.predict(X_test)
#print(y_pred)
plt.scatter(X_train, Y_train , color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('salary vs experience(Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')


plt.scatter(X_test, Y_test , color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('salary vs experience(Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

