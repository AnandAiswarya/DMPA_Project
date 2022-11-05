import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import joblib
import matplotlib.pyplot as plt

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


##
class LogidticRegression:
    def sigmoid(self,z):
        sig = 1/(1+np.exp(-z))
        return sig
    def initialize(self,X):
        weights = np.zeros(X.shape[1]+1,int)
        X = np.c_[np.ones(X.shape[0],int),X]
        return weights,X
    def fit(self,X,y,alpha=0.001,iter=400):
        weights,X = self.initialize(X)
        print(weights)
        def cost(theta):
            z = dot(X,theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y)
            return cost
        cost_list = np.zeros(iter,)
        for i in range(iter):
            weights = weights.astype(float) - alpha*np.dot(X.T,self.sigmoid(np.dot(X,weights))-(y.to_numpy()).reshape(len(y),1))
            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis
##

model = LogidticRegression()
#model = LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
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
age = input("Enter age: ")
openness = input("Enter openness: ")
neuroticism = input("Enter neuroticism: ")
conscientiousness = input("Enter con: ")
agreeableness = input("Enter agree: ")
extraversion = input("Enter extra: ")
result = np.array([gender_no, age, openness,neuroticism, conscientiousness, agreeableness, extraversion], ndmin = 2)
final = scaler.fit_transform(result)
personality = str(show.predict(final)[0])
print("Personality is : ", personality)
