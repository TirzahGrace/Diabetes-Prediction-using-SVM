import numpy as np
from sklearn.model_selection import KFold
from svm import SVM

def cross_validation(X_train_normalized, y_train, n_splits = 5):
    # Split train data into train and validation sets
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("Splits in cross_validation: ",kf.get_n_splits())

    best_accuracy = 0
    best_C = None
    best_learning_rate = None

    for C in [0.01, 0.1, 1, 10]:
        for learning_rate in [0.001, 0.01, 0.1]:
            accuracy_sum = 0

            for train_index, val_index in kf.split(X_train_normalized):
                X_train_fold, X_val_fold = X_train_normalized.iloc[train_index], X_train_normalized.iloc[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                # Train SVM model
                model = SVM(C=C, learning_rate=learning_rate)
                model.fit(X_train_fold, y_train_fold)

                # Evaluate on validation set
                y_pred_val = model.predict(X_val_fold)
                accuracy = np.mean(y_pred_val == y_val_fold)
                accuracy_sum += accuracy

            mean_accuracy = accuracy_sum / kf.get_n_splits()
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_C = C
                best_learning_rate = learning_rate
    
    return best_C, best_learning_rate
                
