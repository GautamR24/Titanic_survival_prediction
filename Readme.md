#  Project: Titanic Survival Prediction
In this project we will try to predict the survival of the passengers.
## Libraries used:
1. Numpy
2. Pandas
3. sklearn
4. matplotlib
5. seaborn
## Dataset:
you can download the dataset from [here](https://www.kaggle.com/c/titanic)
## Code:
The `titanic_survival_prediction.py` contains the code.

## Workflow of code:
1. First importing the dataset.
2. Different features present in dataset are `PassengerId`,`Survival`,`Pclass`,`Name`,`Sex`,`Age`,`Parch`,`Ticket`,`Fare`,`Cabin`,`Embarked`,`SibSp`.
3. finding relation between the survival and different features so that we can drop unnecessary features to train the model accurately.
4. `Sex`,`Pclass`(people with higher socio-economic background),`Parch` are important features.
5. for the `Age` feature : we divided the ages into different age groups, using plots it turned out that child are more likely to survive than any other age group.
6. `Cabin` and `Ticket` feature are not giving any important info therefore dropping them.
7. checking for the missing values in the dataset.
8.  There are missing values in `Age` and `Embarked` feature.
9. In `Embarked` feature there are only `2` missing values, so , it is simply replaced with the value present in majority.
10. The missing value in `Age` feature are `177`, therefore we cant perform the same operation as performed in `Embarked` feature:
    * first, extracted the title form `Name` feature.
    * replaced various titles with more common names, to make the categorization more easy.
    * gave numerical value to different titles.
    * then took the mode age of each age-group and inserted this mode age into the respective age-group whose values are missing.
11. mapped each age-group to a numerical value.
12. Then dropped `Name` feature as now title have been extracted.
13. mapped the values of `Sex` and `Embarked` feature with numerical values.
14. trained different models ,predicted the survival and checked the accuracy.

## Methodology:
1. import the dataset.
2. do the feature extraction, which will help in finding important features.
3. drop unnecessary features.
4. find the features which have missing value.
5. fill the missing values.
6. change all the values of the different features to numerical values so that we can train the model.
7. split the dataset into testing and training dataset.
8. Train and test the model.
9. find the accuracy.


## Result: 
Random Forest gives an Accuracy of **81%**.
