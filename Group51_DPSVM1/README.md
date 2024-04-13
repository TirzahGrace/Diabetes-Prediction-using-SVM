# Diabetes Prediction using SVMs

## Overview:
This project aims to predict the onset of diabetes in patients using Support Vector Machines (SVMs). It involves steps such as data preprocessing, model implementation, hyperparameter tuning, and evaluation.

## Files:

- **src/main.py:** Entry point of the program. It orchestrates the execution of preprocessing, hyper-parameter tuning using cross-validation, model training, and evaluation.
- **src/pre_process.py:** Contains functions for data preprocessing, including splitting the dataset into train and test sets and normalizing the features.
- **src/svm.py:** Defines the SVM class, which implements the Stochastic Gradient Descent algorithm for training an SVM model.
- **src/cross_validation.py:** Implements cross-validation to tune hyperparameters such as C and learning rate.
- **data/diabetes.csv:** Dataset used for training and testing the SVM model.
- **output/output.txt:** Output file containing the results and performance metrics of the SVM model.
- **requirements.txt:** Contains a list of all Python packages and their versions required to run the project.
  
## Usage:

1. Clone the repository:
   ```
   git clone https://github.com/TirzahGrace/Diabetes-Prediction-using-SVM.git
   ```
2. Navigate to the project directory:
   ```
   cd Diabetes-Prediction-using-SVM/Group51_DPSVM1/
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```
   cd src
   python3 main.py > ../output/output.txt 
   ```
   - Results are stored in Group51_DPSVM1/output/output.txt file
## Author:
  - Tirzah Grace (21CS10071)
