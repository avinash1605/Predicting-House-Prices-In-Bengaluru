#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#dataset import and spliting it into X and Y
dataset = pd.read_csv('house_prices.csv')
X= dataset.iloc[:,[0,2,3,5,6,7]].values
Y=dataset.iloc[:,8].values

#removing missing values
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values="NaN",strategy="mean",axis=0)
imp.fit(X)
X=imp.transform(X)

#LABEL ENCODING
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labenc_X = LabelEncoder()
#X[1]=X[1].astype('str')
#X[1]=labenc_X.fit_transform(X[1].astype(str))
X[:,0]=labenc_X.fit_transform(X[:,0])
X[:,1]=labenc_X.fit_transform(X[:,1])
X[:,2]=labenc_X.fit_transform(X[:,2])

#Splitting data sets into training and testing sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,random_state=0)#preferably random state value is used as 0

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train , Y_train)

#predicting
Y_pred_train = regressor.predict(X_train) 
Y_pred_test = regressor.predict(X_test) 

#evaluating accuracy with RMSE

def rmse(Y_pred,Y_test):
    error = np.square(np.log10(Y_pred+1)-np.log10(Y_test +1 )).mean()**0.5
    acc=1-error
    return acc
print ("accuracy on training set = ",rmse(Y_pred_train))
print ("accuracy on test set = ",rmse(Y_pred_test))

