"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Import datasets, classifiers and performance metrics
from sklearn import metrics
from utils import preprocess_data, make_param_combinations, read_digits,predict_and_eval,train_test_dev_split, tune_hparams, train_model
from pandas import DataFrame, set_option

set_option('display.max_columns', 40)

gamma_range = [0.0001, 0.0005, 0.001, 0.01, 0.1]
C_range = [0.1,1,10,100]
svm_param_list_dict = {'gamma':gamma_range,'C':C_range}

max_depth_range = [5,10,20,50,100]
tree_param_list_dict = {'max_depth':max_depth_range}

logistic_solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
logistic_param_list_dict= {'solver': logistic_solvers}

model_type_param_list_dict = {"svm":svm_param_list_dict,"tree":tree_param_list_dict, "lr":logistic_param_list_dict}
# model_type_param_list_dict = {"lr":logistic_param_list_dict}

# test_size_ranges = [0.1, 0.2, 0.3]
# dev_size_ranges = [0.1, 0.2, 0.3]
test_size_ranges = [0.2]
dev_size_ranges = [0.2]
split_size_list_dict = {'test_size':test_size_ranges,'dev_size':dev_size_ranges}

# Read digits: Create a classifier: a support vector classifier
x_digits,y_digits = read_digits()

#Find best model for given gamma and C range

splits = make_param_combinations(split_size_list_dict)
# split = {'test_size':0.2,'dev_size':0.2}
iterations = 1
for split in splits:
    results = []
    for run in range(iterations):
        # Data splitting: Split data into train, test and dev as per given test and dev sizes
        X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(x_digits,y_digits,shuffle=True, **split)

        # Data preprocessing
        X_train = preprocess_data(X_train)
        X_test = preprocess_data(X_test)
        X_dev = preprocess_data(X_dev)

        for model_type in model_type_param_list_dict:
            # print(f"Current model: {model_type}")
            best_hparams,best_model, best_accuracy,best_model_path =  tune_hparams(X_train,y_train,X_dev,y_dev,model_type_param_list_dict[model_type],model_type=model_type)
            _,train_acc = predict_and_eval(best_model,X_train,y_train,c_report=False,c_matrix=False)
            _,test_acc = predict_and_eval(best_model,X_test,y_test,c_report=False,c_matrix=False)
            _,dev_acc = predict_and_eval(best_model,X_dev,y_dev,c_report=False,c_matrix=False)

            # print("Test size=%g, Dev size=%g, Train_size=%g, Train_acc=%.4f, Test_acc=%.4f, Dev_acc=%.4f" % (split['test_size'],split['dev_size'],1-split['test_size']-split['dev_size'],train_acc,test_acc,dev_acc) ,sep='')
            current_run_results = [model_type,run,train_acc,dev_acc,test_acc,str(split),str({x:best_hparams[x] for x in model_type_param_list_dict[model_type]})]
            results.append(current_run_results)
        # print("Best hparams:= ",dict([(x,best_hparams[x]) for x in param_list_dict.keys()]))

        # Getting model predictions on test set
        # Predict the value of the digit on the test subset

        # predicted,best_accuracy = predict_and_eval(best_model,X_test,y_test,c_report=False,c_matrix=False)
        # print("Test Accuracy:",best_accuracy)

header = ["Model_type","Run","train_acc","dev_acc","test_acc","split","params"]
results_df = DataFrame(results,columns=header)
stats = results_df.groupby("Model_type")
print(results_df)
print(stats["test_acc"].agg(['mean', 'std']))
