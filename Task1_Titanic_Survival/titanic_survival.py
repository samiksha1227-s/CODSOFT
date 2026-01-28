import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv")
print(data.head())

data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

X = data.drop('Survived', axis=1)
y = data['Survived']
#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,random_state=42
)

model = LogisticRegression(max_iter = 200)
model.fit(X_train, y_train)

#Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: ", accuracy)
