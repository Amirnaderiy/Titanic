# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:27:42 2023

@author: Asus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# Reading train and test data
train = pd.read_csv("E:/Datasets/Kaggle/Titanic/train.csv")
test = pd.read_csv("E:/Datasets/Kaggle/Titanic/test.csv")

print("Train Shape:",train.shape)
print("Test Shape:",test.shape)

"""
Survived: 0 = No, 1 = Yes

pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd

sibsp: of siblings / spouses aboard the Titanic

parch: of parents / children aboard the Titanic

ticket: Ticket number

cabin: Cabin number

embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
"""




### Feature engineering ###

##Data Clearning

#age
def clearning_age (data):
    data['Age'].fillna(data['Age'].median(), inplace = True)
    
clearning_age(train)
clearning_age(test)


#emarked
def clearning_embarked(data):
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)

clearning_embarked(train)
clearning_embarked(test)

# delete PassengerId, Cabin, Ticket
drop_column = ['PassengerId','Cabin', 'Ticket']

def delete_col(data):
    data.drop(drop_column, axis=1, inplace=True)
    
delete_col(train)
delete_col(test)



##Generate new features

#generate new feaure:  Familysize 
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
    
# generate new feature: Alone

def GenerateF_Alone(data): 
  data['Alone'] = 0
  data.loc[data['FamilySize'] == 1, 'Alone'] = 1
    
GenerateF_Alone(train)
GenerateF_Alone(test)

def GenerateF_LargeFamily(data): 
  data['Alone'] = 0
  data.loc[data['FamilySize'] == 1, 'Large_fam'] = 1
    
GenerateF_Alone(train)
GenerateF_Alone(test)
    

#Categorical Age (Creating 4 columns for Age)
def GenerateF_Child(data):
    data['child'] = 0
    data.loc[data['Age']<=16, 'child']=1
GenerateF_Child(train)
GenerateF_Child(test)

def GenerateF_Old(data):
    data['old'] = 0
    data.loc[data['Age']> 64, 'old']=1
GenerateF_Old(train)
GenerateF_Old(test)  

def GenerateF_youngAdult(data):
    data['young adult'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'young adult'] = 1
GenerateF_youngAdult(train)
GenerateF_youngAdult(test)        
    
def GenerateF_Adult(data):
    data['adult'] = 0
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'adult'] = 1   
GenerateF_Adult(train)
GenerateF_Adult(test)

def GenerateF_MiddleAge(data):
    data['middle_age'] = 0
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'middle_age'] = 1   
GenerateF_MiddleAge(train)
GenerateF_MiddleAge(test)




plt.bar(x=train["Survived"], height=train["child"])
plt.title("Children casualties")
plt.legend()

def drop_age(data):   #remove age column
    data.drop('Age', axis=1, inplace=True)
drop_age(train)
drop_age(test)    


# Category Sex
def sex_cat (data):
    data["Sex"] = data["Sex"].astype("category")
    data["Sex"].cat.categories = [0,1]
    data["Sex"] = data["Sex"].astype("int")
sex_cat(train)
sex_cat(test)

# Category Embarked
def embarked_cat (data):                                   
    data["Embarked"] = data["Embarked"].astype("category")
    data["Embarked"].cat.categories = [0,1,2]
    data["Embarked"] = data["Embarked"].astype("int")
embarked_cat(train)
embarked_cat(test)

#preprocessing for Name
def preprocess_name(data):
    data["Name"]= data["Name"].split(",")
    

#splitting data and label
train_label = train.loc[:,["Survived"]]
train_data= train.copy()                                                                                          
train_data.drop('Survived', axis=1, inplace=True)
   
    

    



