from flask import Flask, request
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op" : "Hello, World POST " + request.json["suffix"]}

@app.route("/sum/<x>/<y>")
def sum(x,y):
    return str(int(x) + int(y))

def load_model():
    model_svm = joblib.load("M23CSA010_svm_gamma:0.001_C:1.joblib")
    model_tree = joblib.load("M23CSA010_tree_max_depth:20.joblib")
    model_lr = joblib.load("M23CSA010_lr_solver:lbfgs.joblib")
    return model_svm,model_tree,model_lr

@app.route("/model/<type>", methods=['POST'])
def pred_model(type):
    if(type=='svm'):
        model = load_model()[0]
    elif(type=='tree'):
        model = load_model()[1]
    elif(type=='lr'):
        model = load_model()[2]
    else:
        return "Invalild model type"
    
    js = request.get_json()
    image = np.array(js['image'])
    image.reshape(1,-1)
    # model = joblib.load('models/svm_gamma:0.001_C:1.joblib')
    predict = model.predict([image])
    return str(predict)

@app.route("/compare", methods=['POST'])
def compare_images():
    js = request.get_json()
    image1 = np.array(js['image1'])
    image2 = np.array(js['image2'])

    image1.reshape(1, -1)
    image2.reshape(1, -1)
    model = joblib.load('models/svm_gamma:0.001_C:1.joblib')
    predict = model.predict([image1,image2])
    # predict2 = model.predict(image2)

    if( (predict[0] == predict[1])):
        return f"True"
    else:
        return f"False"


if __name__ == "__main__":
    app.run(host="0.0.0.0")