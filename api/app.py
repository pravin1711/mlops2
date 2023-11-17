from flask import Flask, request
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/hello/<user>")
def hello_world(user):
    return f"<p>Hello {user}!</p>"

@app.route("/sum/<x>/<y>")
def sum(x,y):
    return str(int(x) + int(y))

@app.route("/model", methods=['POST'])
def pred_model():
    js = request.get_json()
        
    image = np.array(js['image']).reshape(1, -1)
    model = joblib.load('Models/svm_gamma:0.0005_C:1.joblib')
    predict = model.predict([image])
    return str(predict)

@app.route("/compare", methods=['POST'])
def compare_images():
    js = request.get_json()
    image1 = np.array(js['image1'])
    image2 = np.array(js['image2'])

    image1.reshape(1, -1)
    image2.reshape(1, -1)
    model = joblib.load('Models/svm_gamma:0.0005_C:1.joblib')
    predict = model.predict([image1,image2])
    # predict2 = model.predict(image2)

    if( (predict[0] == predict[1])):
        return f"True"
    else:
        return f"False"