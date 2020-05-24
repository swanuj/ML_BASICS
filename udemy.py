import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.impute import SimpleImputer


dataset = pd.read_csv('Data.csv')
print(dataset)

X = dataset.iloc[:,:-1].values

print(">>>>>>>>>>>>>>>>>>>>>")
print(X)
y  = dataset.iloc[:,-1].values

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

print(y)

imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')

imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

print(X)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder() , [0])] , remainder = 'passthrough')

X = np.array(ct.fit_transform(X))

#X = ct.fit_transform(X)

print("encoded>>>>>>>>") 

"""
from sklearn.preprocessing import LabelEncoder
ls = LabelEncoder()
X = ls.fit_transform(X[:,0:1])

"""


print(X)

from sklearn.preprocessing import LabelEncoder
ls = LabelEncoder()
y = ls.fit_transform(y)

print(y)

from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X,y ,test_size = 0.20 , random_state = 1)

print("AAAAAAAAAAAAAAAA")

print(X_train)

print("BBBBBBBBBBBBBBBBBBBBBBBBB") 


print(X_test)

print("cccccccccccccccccccccccccccccccc") 

print(y_train)


print("ddddddddddddddddddddddddddddd") 


print(y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:,3:])

print("^^^^^^^^^^^^^^^^^^^^^^^^^^")

print(X_train)

print("<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>")
print(X_test)