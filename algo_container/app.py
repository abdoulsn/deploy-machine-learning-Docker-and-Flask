from flask import Flask, render_template
from model import get_data
import json

app = Flask(__name__)
data = get_data()

@app.route('/', methods = ['POST'])
def dasboard():
    return json.dumps(data)

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html', data=data)

app.run(host='0.0.0.0')

