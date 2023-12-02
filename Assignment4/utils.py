from sklearn import datasets, metrics, svm, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from sklearn.preprocessing import normalize
import itertools

def read_digits():
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target
    return x,y

def split_data(x,y, test_size, shuffle = False, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=test_size, shuffle=shuffle,random_state=random_state)
    return X_train, X_test, y_train, y_test

def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    data = normalize(data)
    return data
    
def train_model(x,y,model_params={},model_type="svm"):
    if model_type == "svm":
        clf = svm.SVC
    if model_type == "tree":
        clf = tree.DecisionTreeClassifier
    if model_type == "lr":
        clf = LogisticRegression
        
    model = clf(**model_params)
    model.fit(x,y)
    return model

def train_test_dev_split(x,y, test_size, dev_size, shuffle = False, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=test_size, shuffle=shuffle,random_state=random_state)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train,y_train, test_size=dev_size/(1-test_size), shuffle=shuffle,random_state=random_state)
    return X_train, X_test, X_dev, y_train, y_test,y_dev

def predict_and_eval(model,X_test,y_test,c_report=True,c_matrix=True):
    predicted = model.predict(X_test)

    if(c_report == True): print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
    )
        
    accuracy = accuracy_score(y_test,predicted)

    if(c_matrix == True):
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)

        # y_true = [] 
        # y_pred = []
        cm = disp.confusion_matrix
        print(cm)

        # for gt in range(len(cm)):
        #     for pred in range(len(cm)):
        #         y_true += [gt] * cm[gt][pred]
        #         y_pred += [pred] * cm[gt][pred]

        # print(
        #     "Classification report rebuilt from confusion matrix:\n"
        #     f"{metrics.classification_report(y_true, y_pred)}\n"
        # )

    return predicted, accuracy

def make_param_combinations(param_list_dict):
    hparams = param_list_dict.keys()
    ranges = [param_list_dict[x] for x in hparams]
    list_of_all_param_combination=[ dict(zip(hparams,x)) for x in list(itertools.product(*ranges))]
    return list_of_all_param_combination    
        

def tune_hparams(X_train, y_train, X_dev, y_dev, param_list_dict,model_type="svm",c_report=False):
    list_of_all_param_combination = make_param_combinations(param_list_dict)
    best_accuracy_so_far = -1
    best_model = None
    best_model_path = ""
    for params in list_of_all_param_combination:
        cur_model = train_model(X_train, y_train, model_params=params, model_type=model_type)
        _, cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev,c_report=c_report,c_matrix=False)
        if model_type=='lr':
            cur_model_path = "./models/M23CSA019_{}_".format(model_type) +"_".join(["{}:{}".format(k,v) for k,v in params.items()]) + ".joblib"
            dump(cur_model,cur_model_path)
            print(f"Accuracy of {params['solver']} = {cur_accuracy}")

        if cur_accuracy > best_accuracy_so_far:
            best_accuracy_so_far = cur_accuracy
            best_model = cur_model
            best_model_path = "./models/M23CSA019_{}_".format(model_type) +"_".join(["{}:{}".format(k,v) for k,v in params.items()]) + ".joblib"

    

    best_accuracy = best_accuracy_so_far
    best_hparams = best_model.get_params()
    dump(best_model,best_model_path)
    return best_hparams, best_model, best_accuracy, best_model_path