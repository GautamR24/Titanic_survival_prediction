#  Project: Titanic Survival Prediction
In this project we will try to predict the survival of the passengers.
## Libraries used:
1. pandas
2. sklearn
## Dataset
you can download the dataset from [here](https://www.kaggle.com/c/titanic  )
## Code:
The `titanic_survival_prediction.py` contains the code.

## Workflow of code
1. Import the dataset.
2. Get the idea of the type of data present in the dataset
3. Check for missing values in the dataset.
4. In `Age` there are `177` values missing, `Embarked` has `2` values and `Cabin` has more than `400` values missing.
5. On observing out `Age`,`Embarked`,`Cabin`, only `Age` plays an important part in the survival,therefore we will only fill the missing values of the `Age` column.
6. We will create two new objects one having the vital features from the dataset and other having the `survived` column.
7. The important features will be `Age`,`Sex`,`Pclass`,`Fare`.
8. Now the `Sex` feature is a categorical variable having two value in the dataset `male` & `female`,
we will map them with the numerical values so that we can train the model.
9. We will make dictionary like `{"male":0,"female":1}` and then map them with the values of `Sex` column.
10. So all the missing values have been filled and and all columns contain data in numerical form, now we will split the dataset in testing and training dataset.
11. We will fit the model on the training dataset, after training it will tested.
12. At last print the accuracy.

## Result: 
Random Forest gives an Accuracy of **82.2%**.
