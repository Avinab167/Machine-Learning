import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


#Converting categorical data to numerical data
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

X = train[['Pclass','Sex','Age','Fare']].fillna(0)  # features
y = train['Survived']                              # target

model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(test[['Pclass', 'Sex', 'Age', 'Fare']].fillna(0))


output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

