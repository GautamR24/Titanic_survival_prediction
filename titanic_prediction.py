
import pandas as pd
# importing the train dataset
train = pd.read_csv('train.csv')

# getting the idea of the dataset by viewing first few rows
print(train.head())


# checking the missing data
print(pd.isnull(train).sum())

# so there is large amount of data missing in Age and Cabin column and negligible in Embarked column
# from the perspective of survival the Cabin and Embarked column will not play a important role 
# therefore we will concentrate in filling the missing values of Age column
# we will replace the missing values of the Age by taking the median of the column

train["Age"] = train["Age"].fillna(train["Age"].median())
# checking whether the missing values has been replaced or not
print(train["Age"].isnull().sum())

# now by seeing the data i came to the conclusion that the imortant features for survival are Age,Pclass
# Sex and Fare therefore first checking the dependency of these features in survival 
# separating the important columns
train_col = ["Age","Pclass","Sex","Fare"]
test_col = ["Survived"]
imp_column = train[train_col]
# we will test it against the survival column, therefore separating it 
survival_col = train[test_col]

# now the we are taking the Sex column but it is a categorical variable , therefore we will have
# to change it's value with numeric value
# first we will make a dictionary which we will map with it's value
dict = {"male":0,"female":1}
# now maping the values
imp_column["Sex"] = imp_column["Sex"].apply(lambda x:dict[x])

# now we will split the dataaset in training and testing dataset using sklearn
from sklearn.model_selection import train_test_split
imp_column_train,imp_column_test,survival_col_train,survival_col_test = train_test_split(imp_column,survival_col,test_size=0.2,random_state=0)

# now we fill import the randomForest, then train the model, at last test it. 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
randomforest = RandomForestClassifier()
randomforest.fit(imp_column_train, survival_col_train)
y_pred = randomforest.predict(imp_column_test)
acc_randomforest = round(accuracy_score(y_pred, survival_col_test) * 100, 2)
print("the accuracy of random forest is " , acc_randomforest)


