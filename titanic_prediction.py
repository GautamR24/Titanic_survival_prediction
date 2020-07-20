
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
#%matplotlib inline

# importing the train and test dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#temp  = train.describe(include="all")

""" 
 # here just doing data visualizations 
sns.barplot(x="Sex", y="Survived",data=train)

print("percentage of femakes survived",train["Survived"][train["Sex"]== 'female'].value_counts(normalize= True)[1]*100)

print("percentage of males survived",train["Survived"][train["Sex"]== 'male'].value_counts(normalize= True)[1]*100)

sns.barplot(x="Pclass", y="Survived",data=train)

print("percentage of pclass =1 who survived",train["Survived"][train["Pclass"]== 1].value_counts(normalize= True)[1]*100)

print("percentage of pclass =2 who survived",train["Survived"][train["Pclass"]== 2].value_counts(normalize= True)[1]*100)

print("percentage of pclass =3 who survived",train["Survived"][train["Pclass"]== 3].value_counts(normalize= True)[1]*100)

sns.barplot(x="SibSp", y="Survived",data=train)

print("percentage of SibSp = 0 who survived",train["Survived"][train["SibSp"]== 0].value_counts(normalize= True)[1]*100)

print("percentage of pclass =1 who survived",train["Survived"][train["Pclass"]== 1].value_counts(normalize= True)[1]*100)

print("percentage of pclass =2 who survived",train["Survived"][train["Pclass"]== 2].value_counts(normalize= True)[1]*100)

sns.barplot(x="Parch",y="Survived",data=train)  """

# here just trying to get the idea which age group survived the most
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1,0,5,12,18,24,35,60,np.inf]
labels = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup'] = pd.cut(train["Age"],bins,labels=labels)
test['AgeGroup'] = pd.cut(train["Age"],bins,labels=labels)
sns.barplot(x="AgeGroup",y="Survived",data=train)

# observing whetther cabin number has some relation with the survival
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))
print("percentage of CabinBool =1 who survive",train["Survived"][train["CabinBool"]==1].value_counts(normalize=True)[1]*100)
print("percentage of CabinBool =0 who survived",train["Survived"][train["CabinBool"]==0].value_counts(normalize=True)[1]*100)
sns.barplot(x="CabinBool",y="Survived",data=train)

# dropping the unneccasry columns
train = train.drop(['Cabin'],axis=1)
test = test.drop(['Cabin'],axis=1)
train = train.drop(['Ticket'],axis=1)
test = test.drop(['Ticket'],axis=1)


# to replace the missing values in embarked finding which origin is present
# in majority , it may work
print("number of people embarking in southhampton(S):")
southhampton = train[train["Embarked"]=="S"].shape[0]
print(southhampton)

print("number of people embarking in Cherbourg(C):")
cherbourg = train[train["Embarked"]=="C"].shape[0]
print(cherbourg)

print("number of people embarking in Queenstown(Q):")
queenstown = train[train["Embarked"]=="Q"].shape[0]
print(queenstown)

# filling the missing value with "S" as it is present in majority
train = train.fillna({"Embarked":"S"})


combine = [train,test]

# getting the info about various titles present in the dataset
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Aa-z]+)\.',expand=False)
pd.crosstab(train['Title'],train['Sex'])

# since their are numberous titles present in the dataset
# we need to combine some of them so that it becomes easy for processing
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
train[['Title','Survived']].groupby(['Title'],as_index=False).mean()

# giving numerical value to each of the title created above
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# now we will replace missing age with the mode age of their group
# first finding the mode age 
mr_age = train[train["Title"] == 1]["AgeGroup"].mode()
miss_age = train[train["Title"] == 2]["AgeGroup"].mode()
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() 
master_age = train[train["Title"] == 4]["AgeGroup"].mode()
royal_age = train[train["Title"] == 5]["AgeGroup"].mode() 
rare_age = train[train["Title"] == 6]["AgeGroup"].mode() 
# now maping the missing age with age title
age_title_mapping = {1:"Young Adult",2:"Student",3:"Adult",4:"Baby",5:"Adult",6:"Adult"}
train = train.fillna({"Age": train["Title"].map(age_title_mapping)})
test = test.fillna({"Age": test["Title"].map(age_title_mapping)})



# here we are maping age to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)


imputer = SimpleImputer(missing_values=np.nan, strategy="median")
train[['AgeGroup']] = imputer.fit_transform(train[['AgeGroup']])
test[['AgeGroup']] = imputer.fit_transform(test[['AgeGroup']])

# dropping the Age feature
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)

# dropping name feature
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)

# here we are maping sex value to numeric value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)


# here we are maping embarked value to numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)

from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)




from sklearn.metrics import accuracy_score



"""from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
"""
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print("the accuracy of random forest is " , acc_randomforest)

"""from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
"""
"""models = pd.DataFrame({
    'Model': [ 'KNN', 
              'Random Forest',, 
              'Decision Tree'],
    'Score': [acc_knn,acc_randomforest,acc_decisiontree]})
models.sort_values(by='Score', ascending=False)
"""
