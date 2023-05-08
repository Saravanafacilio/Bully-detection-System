#!/bin/env python
from app import create_app, socketio

from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = create_app(debug=True)


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    df = pd.read_csv("twitterdataset.csv", encoding="latin-1")

    # Features and Labels
    df['label'] = df['class'].map({'Non-Bullying': 0, 'Bullying': 1})
    X = df['message']
    y = df['label']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X) # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    #Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)

    data = [message]
    vect = cv.transform(data).toarray()
    prediction = clf.predict(vect)
    return render_template('result.html', prediction=prediction[0])



if __name__ == '__main__':
    socketio.run(app)
