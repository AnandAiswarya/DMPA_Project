import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('train dataset.csv')

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
input_cols = ['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']
output_cols = ['Personality (Class label)']

scaler = StandardScaler()
data[input_cols] = scaler.fit_transform(data[input_cols])
data.head()

X = data[input_cols]
Y = data[output_cols]

model = SVC(kernel = 'linear',gamma = 'scale', shrinking = False,)
model.fit(X, Y)

test_data = pd.read_csv('test dataset.csv')
test_data['Gender'] = le.fit_transform(test_data['Gender'])
test_data[input_cols] = scaler.fit_transform(test_data[input_cols])
X_test = test_data[input_cols]
Y_test = test_data['Personality (class label)']
test_data.head()

y_pred= model.predict(X_test)

print(accuracy_score(Y_test,y_pred)*100)

joblib.dump(model, "train_model.pkl")
show = joblib.load("train_model.pkl")
gender = input("Enter gender: ")
if(gender == "Female"):
    gender_no = 1
else:
    gender_no = 2
age = input("Enter gender: ")
openness = input("Enter openness: ")
neuroticism = input("Enter neuroticism: ")
conscientiousness = input("Enter con: ")
agreeableness = input("Enter agree: ")
extraversion = input("Enter extra: ")
result = np.array([gender_no, age, openness,neuroticism, conscientiousness, agreeableness, extraversion], ndmin = 2)
final = scaler.fit_transform(result)
personality = str(show.predict(final)[0])
print("Personality is : ", personality)