from flask import Flask

from app.model.thread_model import SentimentModel
from flask import request
import json
from time import time
import tensorflow as tf

app = Flask(__name__)

m = SentimentModel()
m.model = m.load_saved_model()
m.tokenizer = m.load_tokenizer()
m.classes = m.load_classes()


@app.route('/sentiment/reco', methods=['POST'])
def predict():
    start = time()
    res = {}
    content = request.json
    text = content['text']
    pred = m.predict(text)
    res['data'] = pred
    res['elapsed_time'] = time() - start
    return json.dumps(res)
