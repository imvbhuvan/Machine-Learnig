import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt


data = pd.read_csv(r"D:\ML\ML-codes\My_Repo\Datasets\heart_disease.csv")

#Checking the infomartion about the dataset
data.info()

#Checking if there are any null values in the given dataset
data.isnull().sum()

#Segregating the categorical and numerical feature columns
data_cat = data[['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking','Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth','Asthma', 'KidneyDisease', 'SkinCancer']]

data_num = data[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']]


for i in data_num.columns:
    sns.boxplot(x=data_num[i])
    plt.show()
plt.show()

plt.figure(figsize=(15,6))
sns.countplot(y=data['Sex'],hue='HeartDisease', data=data)
plt.xticks(rotation = 0)
plt.show()

plt.figure(figsize=(15,6))
sns.countplot(y=data['Smoking'],hue='HeartDisease',data=data)
plt.xticks(rotation = 0)
plt.show()

plt.figure(figsize=(15,6))
sns.countplot(y=data['Stroke'],hue='HeartDisease',data=data)
plt.xticks(rotation = 0)
plt.show()

plt.figure(figsize=(20,6))
sns.countplot(y=data['AgeCategory'],hue='HeartDisease',data=data)
plt.xticks(rotation = 0)
plt.show()

plt.figure(figsize=(20,6))
sns.countplot(y=data['Race'],hue='HeartDisease',data=data)
plt.xticks(rotation = 0)
plt.show()

plt.figure(figsize=(20,6))
sns.countplot(y=data['Race'],data=data)
plt.xticks(rotation = 0)
plt.show()

plt.figure(figsize=(16,6))
sns.countplot(y=data['GenHealth'],hue='HeartDisease',data=data)
plt.xticks(rotation = 0)
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(y=data['PhysicalHealth'],hue='Sex',data=data)
plt.xticks(rotation = 0)
plt.show()


plt.figure(figsize=(10,6))
sns.countplot(y=data['MentalHealth'],hue='Sex',data=data)
plt.xticks(rotation = 0)
plt.show()

#Performing the label encoding operations for the categoriacl variables
for i in data_cat.columns:
    le=LabelEncoder()
    label=le.fit_transform(data_cat[i])
    data_cat[i]=label


data_cat.head()

#merging the data
data1=pd.concat([data_cat,data_num],axis=1)

#checking the correlation between columns using the heatmap
plt.figure(figsize=(20,8))
sns.heatmap(data1.corr())

#removing the unsignificant columns after the EDA
data1.drop(['Race','BMI'],axis=1,inplace=True)

#preparing data for splitting into train and test sets
X=data1.iloc[:,1:]
y=data1.iloc[:,0]

#splitting up the data to train and test sets using the sklearn function
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

from sklearn.linear_model import LogisticRegression

#applying the log reg model 
model=LogisticRegression()
model.fit(x_train,y_train)

#making predictions on the test set
y_pred = model.predict(x_test)

print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

from sklearn.metrics import accuracy_score,confusion_matrix

#checking the Confusion Matrix for the model
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
        
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

#checking the precision, recall and F1 score of the model
print('Precision: %.3f' % precision_score(y_test, y_pred))	
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))


#printing the number of missclassified samples in the test set
print('Misclassified samples: %d' % (y_test != y_pred).sum())
