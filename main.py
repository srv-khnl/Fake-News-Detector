import numpy as np
import pandas as pd
import sklearn
import ast
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

true_df= pd.read_csv("True.csv")
fake_df= pd.read_csv("Fake.csv")

#data preprocessing
true_df= true_df[["text"]]
fake_df=fake_df[["text"]]
true_df["value"] = 1
fake_df["value"] = 0
merged_df = pd.concat([true_df, fake_df], ignore_index=True)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

merged_df["text"]=merged_df["text"].apply(preprocess)
X=merged_df["text"]
y=merged_df["value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(X_train)
xv_test = vectorization.transform(X_test)
lr = LogisticRegression()
lr.fit(xv_train, y_train)
pred_lr= lr.predict(xv_test)
rf = RandomForestClassifier()
rf.fit(xv_train, y_train)
predict_rf= rf.predict(xv_test)

def predict_news():
    user_input= input("Enter the news text")
    processed_user_input= preprocess(user_input)
    vectorized_user_input= vectorization.transform([processed_user_input])
    answer(lr, vectorized_user_input)
    answer(rf, vectorized_user_input)

def answer(model, vectorized_user_input):
    prediction = model.predict(vectorized_user_input)
    if prediction == 1:
        print(f'The News is True as per {model.__class__.__name__}')
    else:
        print(f'The News is Fake as per {model.__class__.__name__}')

predict_news()

