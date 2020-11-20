# -*- coding: utf-8 -*-
"""sgd_lr.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D8QXagTLxZueEMOluIId8E4GxkP47w28
"""

import warnings
import time 
from mpi4py import MPI 
warnings.filterwarnings("ignore")
# from sklearn.datasets import load_boston
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from numpy import random
from sklearn.model_selection import train_test_split

from google.colab import drive

start_time = time.time()
drive.mount("/content/drive")
data = '/content/drive/Shared drives/Sci_Com/rideshare_kaggle.csv'



# data pre-processing

import pandas as pd
import numpy as np
df = pd.read_csv(data)
#print(df.head(1))
#print(df.iloc[0,0:57])
#print(df.columns.values)
a = np.unique(df[['source', 'destination']].values)

locations_list = ['Back Bay','Beacon Hill','Boston University','Fenway',
 'Financial District','Haymarket Square','North End','North Station',
 'Northeastern University','South Station','Theatre District','West End']

# create dictionary mapping locations to variables
locations_map = {}
k = 1
for i in range(0, len(locations_list)):
  locations_map[locations_list[i]] = k
  k += 1

# replace locations wityh variables
df['source'] = df['source'].map(locations_map)
df['destination'] = df['destination'].map(locations_map)

short_summary_list = [' Clear ',' Drizzle ', ' Foggy ', ' Light Rain ', ' Mostly Cloudy ',' Overcast ' ,' Partly Cloudy ' ,' Possible Drizzle ' ,' Rain ']

# create dictionary mapping locations to variables
summary_map = {}
k = 1
for i in range(0, len(short_summary_list)):
  summary_map[short_summary_list[i]] = k
  k += 1

df['short_summary'] = df['short_summary'].map(summary_map)

#map cab-type to variables
df['cab_type'] = df['cab_type'].map({'Lyft':0, 'Uber':1})

#map name to variables
cab_names_list = ['Black','Black SUV','Lux','Lux Black','Lux Black XL','Lyft','Lyft XL', 'Shared','Taxi','UberPool','UberX','UberXL','WAV']
names_map = {}
k = 1
for i in range(0, len(cab_names_list)):
  names_map[cab_names_list[i]] = k
  k += 1

df['name'] = df['name'].map(names_map)


# drop unwanted columns
# df = df.drop(['product_id', 'timezone', 'id', 'icon', 'long_summary', 'datetime', 'timestamp'], axis=1)
df = df[df.columns[1:20]]
df = df.drop(['product_id', 'timezone', 'datetime', 'apparentTemperature','temperature'], axis=1)
print(df.columns.values)

# replace null values by mean price
df['price'] = df['price'].fillna(value=df['price'].mean())

'''
for i in range(0,len(df.columns)):
  print(i)
  print(df.columns[i])
  print(df[df.columns[i]].isna().sum())
'''

#boston_data=pd.DataFrame(load_boston().data,columns=load_boston().feature_names)
Y = df['price']
X = df.drop(['price'], axis=1)

'''
print(np.all(np.isfinite(X)))
 print(np.any(np.isnan(X)))

print(np.all(np.isfinite(Y)))
 print(np.any(np.isnan(Y)))
'''
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)

'''
print("X Shape: ",X.shape)
print("Y Shape: ",Y.shape)
print("X_Train Shape: ",x_train.shape, x_train.isna().sum())
print("X_Test Shape: ",x_test.shape, x_test.isna().sum())
print("Y_Train Shape: ",y_train.shape, y_train.isna().sum())
print("Y_Test Shape: ",y_test.shape, y_test.isna().sum())

'''

# standardizing data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


## Adding the PRIZE Column in the data
train_data = pd.DataFrame(x_train)
train_data['price'] = y_train.tolist()

#print(len(train_data), len(train_data.columns))


x_test=np.array(x_test)
y_test=np.array(y_test)

print(np.all(np.isfinite(x_train)))
 print(np.any(np.isnan(x_train)))

print(np.all(np.isfinite(y_train)))
 print(np.any(np.isnan(y_train)))

print(np.all(np.isfinite(x_test)))
 print(np.any(np.isnan(x_test)))

print(np.all(np.isfinite(y_test)))
 print(np.any(np.isnan(y_test)))

def MyCustomSGD(train_data,learning_rate,n_iter,k,divideby):
    
    # Initially we will keep our W and B as 0 as per the Training Data
    w=np.zeros(shape=(1,train_data.shape[1]-1))
    b=0
    
    cur_iter=1
    while(cur_iter<=n_iter): 

        # We will create a small training data set of size K
        temp=train_data.sample(k)
        
        # We create our X and Y from the above temp dataset
        y = np.array(temp['price'])
        x = np.array(temp.drop('price',axis=1))

        # We keep our initial gradients as 0
        w_gradient=np.zeros(shape=(1,train_data.shape[1]-1))
        b_gradient=0
        
        for i in range(k): # Calculating gradients for point in our K sized dataset
            prediction=np.dot(w,x[i])+b
            w_gradient=w_gradient+(-2)*x[i]*(y[i]-(prediction))
            b_gradient=b_gradient+(-2)*(y[i]-(prediction))
        
        #Updating the weights(W) and Bias(b) with the above calculated Gradients
        w=w-learning_rate*(w_gradient/k)
        b=b-learning_rate*(b_gradient/k)
        
        # Incrementing the iteration value
        cur_iter=cur_iter+1
        
        #Dividing the learning rate by the specified value
        #learning_rate=learning_rate/divideby
        
    return w,b #Returning the weights and Bias

def predict(x,w,b):
    y_pred=[]
    for i in range(len(x)):
        y=np.asscalar(np.dot(w,x[i])+b)
        y_pred.append(y)
    return np.array(y_pred)

w,b=MyCustomSGD(train_data,learning_rate=0.01,n_iter=1000,divideby=2,k=500)
#y_train_customsgd=predict(x_train,w,b)
y_pred_customsgd=predict(x_test,w,b)

plt.scatter(y_test,y_pred_customsgd)
#plt.scatter(y_train,y_train_customsgd)
plt.grid()
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Scatter plot from actual y and predicted y')
plt.show()
print('Mean Squared Error :',mean_squared_error(y_test, y_pred_customsgd))
total_time = time.time()-start_time 
print("Total time taken:", total_time)
#print('Mean Squared Error :',mean_squared_error(y_train, y_train_customsgd))
