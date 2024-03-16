import pandas as pd
from sklearn.model_selection import train_test_split


def pre_process(df):

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Outcome']), df['Outcome'], test_size=0.2, random_state=42)

    # Compute mean and variance over train set
    mean_train = X_train.mean()
    std_train = X_train.std()

    # Normalize train set
    X_train_normalized = (X_train - mean_train) / std_train

    # Normalize test set using mean and variance from train set
    X_test_normalized = (X_test - mean_train) / std_train

    y_train = y_train.values
    y_test = y_test.values

    return X_train_normalized, X_test_normalized, y_train, y_test