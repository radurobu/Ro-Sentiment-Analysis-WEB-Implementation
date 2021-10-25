from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
#import joblib
import pandas as pd
import torch
from torch import tensor, device, dtype, nn, save, load, set_num_threads
#from keras.models import model_from_json
import torch.nn.functional as F
import gc
import importlib

set_num_threads(1) #F important, fara a seta numara de thred-uri modelul nu functioneaza


app = Flask(__name__)

######## Preparing the Classifier (Loading pre-trained models)
cur_dir = os.path.dirname(__file__)
#cur_dir = 'C:/Users/Radu/Desktop/ML Projects/Sentiment Analysis WEB Implemenation/'
#os.chdir(cur_dir)

model = load(os.path.join(cur_dir,"pkl_objects","bert-base-romanian-cased-v1.pth"))
model.eval()
tokenizer = load(os.path.join(cur_dir,"pkl_objects","tokenizerbert-base-romanian.pth"))

def swish(x):
    return x * F.sigmoid(x)

#F.relu()
class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(768, 768)
        self.b1 = nn.BatchNorm1d(768)
        self.fc2 = nn.Linear(768, 576)
        self.b2 = nn.BatchNorm1d(576)
        self.fc3 = nn.Linear(576,384)
        self.b3 = nn.BatchNorm1d(384)
        self.fc4 = nn.Linear(384,3)
        self.out = nn.Softmax()

    def forward(self,x):

        x = swish(self.fc1(x))
        x = self.b1(x)
        x = swish(self.fc2(x))
        x = self.b2(x)
        x = swish(self.fc3(x))
        x = self.b3(x)
        x = self.out(self.fc4(x))

        return x


classifier = Network()
classifier.load_state_dict(load(os.path.join(cur_dir,"pkl_objects","model_sentiment_pytorch.pth")))
classifier.eval()

db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(document):
    label = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    document = document.lower() # Transform text to lowercase
    tokens = tensor(tokenizer.encode(document, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(tokens)
    probs = classifier(torch.from_numpy(outputs[1].detach().numpy()))
    y = np.argmax(probs.detach().numpy())
    proba = np.max(probs.detach().numpy())
    #y = 1
    #proba = 1
    return label[y], proba
'''
def classify(document):
    document = document.lower() # Transform text to lowercase
    tokens = tensor(tokenizer.encode(document, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(tokens)
    probs = 1 - classifier.predict_proba(outputs[1].detach().numpy())
    label = np.select([0<= probs < 0.45, 0.45<= probs < 0.65, 0.65<= probs <= 1], ['Positive', 'Neutral', 'Negative'], default='Unknown')
    if label == 'Positive':
        prob = 1 - probs[0][0]
    else: prob = probs[0][0]
    return label[0][0], prob'''

# Da eroare daca nu initializezi cu un exemplu
text='Acceptabil dar nu ma da pe spate'
label, prob = classify(text)
print(label, prob)

def train(document, y):
    document = document.lower() # Transform text to lowercase
    X = tensor(tokenizer.encode(document, add_special_tokens=True)).unsqueeze(0)
    classifier.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

######## Flask
class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=10)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        print( 'we are at classification')
        y, proba = classify(review)
        print( 'we passed classification')
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'Negative': 0, 'Positive': 1, 'Neutral': 2}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = -1
    #train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=False)
