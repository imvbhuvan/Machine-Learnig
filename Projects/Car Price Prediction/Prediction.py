import os 
import numpy as np 
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt6




os.chdir(r'D:\ML\Datasets')
data=pd.read_csv('cars_sampled.csv',na_values=['?','??','?????'])
print(data.head(5))
print(data.isnull().sum())

cars = data.copy()
cars.info()
cars.describe()
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', 500)
cars.describe()

# Dropping the Unecessary columns
col = ['name','lastSeen','postalCode','dateCrawled','dateCreated']
cars = cars.drop(columns=col , axis = 1)
print(cars)

cars.drop_duplicates(keep='first', inplace=True)
print(cars.isnull().sum())
yw = cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration'] > 2018)
sum(cars['yearOfRegistration'] < 1950)
sns.regplot(x='yearOfRegistration',y='price',scatter= True,fit_reg=False,data = cars)
 # working range in b/w  1950 and 2018
 
pc =  cars['price'].value_counts().sort_index()
print(pc)
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(cars['price'])
sum(cars['price'] > 150000)
sum(cars['price'] < 100 )
# working range in b/w 100 and 150000

pow =  cars['powerPS'].value_counts().sort_index()
print(pow)
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter= True,fit_reg=True,data = cars)
sum(cars['powerPS'] > 500)
sum(cars['powerPS'] < 10 )
# working range in bw 500 and 10

# Setting the working range of data 
cars = cars[(cars.yearOfRegistration <=2018) & (cars.yearOfRegistration >=1950) & (cars.price <= 150000) & (cars.price >= 100) & (cars.powerPS <= 500) & (cars.powerPS >= 10)]
# ~6700 records are dropped 

cars['monthOfRegistration']/=12
 
# Creating a new column of Age of the car
cars['Age'] = (2018-cars['yearOfRegistration']+cars['monthOfRegistration'])
cars['Age'] = round(cars['Age'],2)
cars.describe()

cars = cars.drop(columns=['yearOfRegistration','monthOfRegistration'] )

# Visualizing the parameters 

sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])


 # Age vs Price
sns.set(style='darkgrid') 
sns.regplot(x='Age',y='price',scatter= True,fit_reg=False,data = cars)

# Power vs Price 
sns.regplot(x='powerPS',y='price',scatter= True,fit_reg=False,data = cars)

# Check all the remaining columns which affect the price and drop the insiginficant one
# Drop the offertype , abtest and seller colums

cars = cars.drop(columns=['offerType' , 'abtest' , 'seller'] )


cars_copy= cars.copy()

# Checking the correlation 

cars1 = cars.select_dtypes(exclude=[object])
print(cars1.corr())

# Omitting the empty rows
cars_omit = cars.dropna(axis=0)


# Get rid of the strings

cars_omit = pd.get_dummies(cars_omit, drop_first=True)

# Model building with the omitted data

x1 = cars_omit.drop(['price'], axis='columns',inplace=False)
y1 = cars_omit['price']

# Transforming price as the log value for better predictions
y1 = np.log(y1)

# Splitting the data for model buildingg

xtrain , xtest , ytrain , ytest = train_test_split(x1,y1 , test_size=0.3 , random_state=3)
print(xtrain.shape , ytrain.shape , xtest.shape , ytest.shape)

# Building the Base model from the omitted data
# Base model is used for comparison

bp = np.mean(ytest)
print(bp)

bp = np.repeat(bp , len(ytest))

baserm = np.sqrt(mean_squared_error(ytest , bp))
print(baserm)
 
# Any model we build should give  an error less than this 
# Then our model is quite good

#Linear regression with Omitted Data

lgr = LinearRegression(fit_intercept=True)

model=lgr.fit(xtrain,ytrain)

pred = lgr.predict(xtest)

mse = mean_squared_error(ytest, pred)
rmse = np.sqrt(mse)
print(rmse)
print(model.score(xtrain,ytrain))
print(model.score(xtest,ytest))

# Finding the residuals 
resi = ytest - pred
sns.regplot(x=ytest,y=resi,fit_reg=(False),scatter=True,data=cars)
resi.describe()


# Random Forest with omitted data 

rf = RandomForestRegressor(n_estimators=100,random_state=1,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4)

modelrf = rf.fit(xtrain,ytrain)

predrf = rf.predict(xtest)

mserf = mean_squared_error(ytest, predrf)
rmserf = np.sqrt(mserf)
print(rmserf)

print(modelrf.score(xtrain,ytrain))
print(modelrf.score(xtest,ytest))

# Model Building imputed data 

carsimp = cars.apply(lambda x:x.fillna(x.median()) if x.dtype == 'float' else x.fillna(x.value_counts().index[0]))
   
carsimp.isnull().sum()

carsimp = pd.get_dummies(carsimp,drop_first=True)


x2 = carsimp.drop(['price'], axis='columns',inplace=False)
y2 = carsimp['price']

y2 = np.log(y2)

# Splitting

xtrain1 , xtest1 , ytrain1 , ytest1 = train_test_split(x2,y2 , test_size=0.3 , random_state=3)
print(xtrain1.shape , ytrain1.shape , xtest1.shape , ytest1.shape)

# Linear regression for imputed data 

lgr1 = LinearRegression(fit_intercept=(True))

model1 = lgr1.fit(xtrain1,ytrain1)

pred1 = lgr1.predict(xtest1)

mse1 = mean_squared_error(ytest1,pred1)
rmse1 = np.sqrt(mse1)
print(rmse1)

resi1 = ytest1 - pred1
sns.regplot(x=ytest1,y=resi1,fit_reg=False)
sns.regplot(x=ytest1,y=pred1,fit_reg=True , marker='*' )
 
print(model1.score(xtrain1,ytrain1))
print(model1.score(xtest1,ytest1)) 

# Random Forest with Imputed Data

rf1 = RandomForestRegressor(n_estimators=100,random_state=1,max_features='auto',max_depth=100,min_samples_split=10,min_samples_leaf=4)

modelrf1 = rf1.fit(xtrain1,ytrain1)

predrf1 = rf1.predict(xtest1)

mserf1 = mean_squared_error(ytest1, predrf1)
rmserf1 = np.sqrt(mserf1)
print(rmserf1)

print(modelrf1.score(xtrain1,ytrain1))
print(modelrf1.score(xtest1,ytest1))
 
print('Regression Scores')
print( rmse)
print( rmse1)
print('\n\n')
print('Random-Forest Scores')
print(rmserf)
print(rmserf1)















