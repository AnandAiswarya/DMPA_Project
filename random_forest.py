import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

#READING AND STANDARDIZING DATA
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

# FITTING MODEL
model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(X, Y)

test_data = pd.read_csv('test dataset.csv')
test_data['Gender'] = le.fit_transform(test_data['Gender'])
test_data[input_cols] = scaler.fit_transform(test_data[input_cols])
X_test = test_data[input_cols]
Y_test = test_data['Personality (class label)']
test_data.head()

y_pred= model.predict(X_test)

# ACCURACY
print(accuracy_score(Y_test,y_pred)*100)

#CONFUSION MATRIX
cm = confusion_matrix(Y, model.predict(X))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

#CLASSIFICATION REPORT
print(classification_report(Y, model.predict(X)))

#DUMP THE MODEL
joblib.dump(model, "train_model.pkl")
show = joblib.load("train_model.pkl")

#INPUTTING VALUES
gender = input("Enter gender: ")
if(gender == "Female"):
    gender_no = 1
else:
    gender_no = 2
age = input("Enter gender: ")
print("Provide your responses on a scale of 1-10 /n")

#CALCULATING OPENNESS SCORE
o_1 = float(input("How likely are you to talk to a friend's friend in a club?"))
o_2 = float(input("How likely are you to try new things at a restaurant?"))
o_3 = float(input("How likely are you to share your problems and secrets with your close friends/family?"))
openness = (o_1+o_2+o_3)/3

#CALCULATING NEUROTICISM SCORE

n_1 = float(input("How often do you experience mood swings and anxiety?"))
n_2 = float(input("How likely are you to get easily offended or hurt?"))
n_3 = float(input("How often do you feel you're lonely or isolated?"))
neuroticism = (n_1+n_2+n_3)/3

#CALCULATING CONSCIENTIOUSNESS SCORE

c_1 = float(input("How likely are you to focus on long term goals than short term ones?"))
c_2 = float(input("How rigid are you about perfection in your tasks/duties?"))
c_3 = float(input("How often to do you cancel plans to complete personal work?"))
conscientiousness = (c_1+c_2+c_3)/3

#CALCULATING AGREEABLENESS SCORE

a_1 = float(input("How likely are you put your others' needs before your own?"))
a_2 = float(input("How emotionally supportive are you towards others?"))
a_3 = float(input("How likely are you to resolve conflict rather than staying out of it?"))
agreeableness = (a_1+a_2+a_3)/3
#CALCULATING EXTRAVERSION SCORE

e_1 = float(input("How much do you prefer having a big group of friends?"))
e_2 = float(input("How likely are you to think out loud as opposed to thinking silently?"))
e_3 = float(input("How much do you enjoy working in teams?"))
extraversion = (e_1+e_2+e_3)/3

#CALCULATING RESULTANT ARRAY
result = np.array([gender_no, age, openness,neuroticism, conscientiousness, agreeableness, extraversion], ndmin = 2)
final = scaler.fit_transform(result)
personality = str(show.predict(final)[0])

#FINAL PERSONALITY PREDICTION
print("Your Personality is : ", personality, "!")