import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('house_prices.csv')

#Removing Location Null
data = data.dropna(axis=0, subset=['location']) #Removing "1" Row which is having location Null as it is irrelevant to have a record for predicting price which has no loacation

#Imputing Nulls From Size With Mode
data['size'].fillna(data['size'].mode()[0], inplace=True)

#Imputation Of Mode Based on Location
def imputer_mode(main_data, od_val, column_name='size'):
    imputer_list=[]
    for i, j in zip(main_data,od_val):
        if(str(i)!='nan'):
            imputer_list.append(i)
        else:
            OriDes_Val=j
            temp=pd.DataFrame()
            temp=data.loc[(data['location']==OriDes_Val) & (~data[column_name].isnull())]
            val=temp[column_name].mode()
            imputer_list.append(val)
    print("Done",str(column_name))
    return imputer_list
#Imputation Of Mode Based on Location
def imputer_mean(main_data, od_val, column_name='size'):
    imputer_list=[]
    for i, j in zip(main_data,od_val):
        if(str(i)!='nan'):
            imputer_list.append(i)
        else:
            OriDes_Val=j
            temp=pd.DataFrame()
            temp=data.loc[(data['location']==OriDes_Val) & (~data[column_name].isnull())]
            val=temp[column_name].mean()
            imputer_list.append(val)
    print("Done",str(column_name))
    return imputer_list

#Mode
data['society']=imputer_mode(data['society'], data['location'], column_name='society')
#Mean
data['total_sqft']=imputer_mean(data['total_sqft'], data['location'], column_name='total_sqft')
data['bath']=imputer_mean(data['bath'], data['location'], column_name='bath')
data['balcony']=imputer_mean(data['balcony'], data['location'], column_name='balcony')

data['bath'].fillna(data['bath'].mean(), inplace=True)
data['balcony'].fillna(data['balcony'].mean(), inplace=True)

data.isnull().sum()

data.head()

#Imputation Done 
# Now Comes Encoding
#5 Columns Need To Be Encoded
#1. We Will Convert the Columns To String Type
#2. Do a Label Encoding
#3. Do a One Hot Encoding
##Conversion to String
data['area_type']=data['area_type'].astype('str')
data['availability']=data['availability'].astype('str')
data['location']=data['location'].astype('str')
data['size']=data['size'].astype('str')
data['society']=data['society'].astype('str')

#Creating Dummies For the Variables
#Dummies For Area Type
e=pd.get_dummies(data['area_type'],prefix="area_type") 
data=pd.concat([data, e], axis=1)
#Dummies For Availiability
e=pd.get_dummies(data['availability'],prefix="availability") 
data=pd.concat([data, e], axis=1)
#Dummies For Location
e=pd.get_dummies(data['location'],prefix="location") 
data=pd.concat([data, e], axis=1)
#Dummies For Size
e=pd.get_dummies(data['size'],prefix="size") 
data=pd.concat([data, e], axis=1)
#Dummies For Society
e=pd.get_dummies(data['society'],prefix="society") 
data=pd.concat([data, e], axis=1)

not_in_list=['area_type','availability','location','size','society','price','total_sqft']
features = data[list(set(data.columns)-set(not_in_list))]
labels=data['price']

from sklearn.cross_validation import train_test_split
train_features,test_features,train_labels,test_labels = train_test_split(features,labels,train_size=0.7,random_state=9)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble  import RandomForestRegressor
from xgboost import XGBRegressor

model=XGBRegressor()
model.fit(train_features.values,train_labels.values)
pred=model.predict(test_features.values)
merror=mean_squared_error(test_labels.values,pred)
print(merror)
print("Root mean Squared error in days for delays --model 1:")
print(merror**.5)




