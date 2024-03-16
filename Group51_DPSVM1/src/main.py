# Group Number: 51
# Roll Number: 21CS10071
# Project Number : 2
# Project Code: DPSVM1
# Project Title: Diabetes Prediction using Support Vector Machines 

import pandas as pd
from pre_process import pre_process
from svm import SVM
from cross_validation import cross_validation
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

def main():
    try:
        file_path = '../data/diabetes.csv'  # path of the data source diabetes.csv
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found. Please make sure the file path is correct.")
        return

    X_train_normalized, X_test_normalized, y_train, y_test  = pre_process(df=df)

    best_C, best_learning_rate = cross_validation(X_train_normalized=X_train_normalized, y_train=y_train)

    print("Best C: ", best_C)
    print("Best Learning rate: ", best_learning_rate)
    final_model = SVM(C = best_C, learning_rate = best_learning_rate)
    final_model.fit(X_train_normalized, y_train)
    print("Final weights (w): ", final_model.w)
    print("Final bias (b): ", final_model.b)

    y_pred_svm = final_model.predict(X_test_normalized)
    print("Results of the SVM Model Implemented: ")
    print(classification_report(y_test, y_pred_svm))

    linear_svc = LinearSVC(max_iter=5000)
    linear_svc.fit(X_train_normalized, y_train)
    y_pred_linear_svc = linear_svc.predict(X_test_normalized)
    print("Results of the Inbuilt Linear SVC: ")
    print(classification_report(y_test, y_pred_linear_svc))



if __name__ == '__main__':
    main()