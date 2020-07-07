from flask import Flask, render_template
import requests
import json

app = Flask(__name__)

data = requests.post(url='http://algo_container:5000/')

@app.route('/')
def index():
    return render_template('index.html', data=json.loads(data.text))

app.run(host='0.0.0.0')
