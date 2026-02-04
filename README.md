ML Assignment 2 – Classification Models & Deployment
##LINK
https://heartdiseasedataset.streamlit.app/#machine-learning-model-evaluation-classification-app

1. Problem Statement

The objective of this project is to implement, evaluate, and deploy multiple machine learning classification models on a real-world dataset. The project demonstrates the complete end-to-end ML workflow including data preprocessing, model training, evaluation using standard metrics, building an interactive web application using Streamlit, and deploying the application on Streamlit Community Cloud.

2. Dataset Description

Dataset Name: Heart Disease Dataset

Source: Public dataset (UCI / Kaggle)

Type: Binary Classification

Number of Instances: 900+

Number of Features: 13 input features

Target Variable:

1 → Presence of heart disease

0 → Absence of heart disease

The dataset contains clinical attributes such as age, sex, cholesterol level, resting blood pressure, maximum heart rate, and other medical indicators. All preprocessing and train–test splitting were performed before model training.

3. Models Implemented

The following six classification models were implemented on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble Model)

XGBoost (Ensemble Model)

4. Evaluation Metrics

Each model was evaluated using the following metrics:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

📊 Model Performance Comparison Table
Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.85	0.88	0.84	0.83	0.83	0.59
Decision Tree	0.97	0.97	0.97	0.97	0.97	0.97
KNN	0.89	0.91	0.88	0.87	0.87	0.67
Naive Bayes	0.80	0.87	0.75	0.89	0.81	0.61
Random Forest	0.98	1.00	1.00	0.97	0.98	0.97
XGBoost	0.98	0.99	1.00	0.97	0.98	0.97
5. Model Performance Observations
Model Name	Observation
Logistic Regression	Performs reasonably well on linearly separable data but shows lower performance compared to ensemble methods.
Decision Tree	Achieves high accuracy but can be prone to overfitting without proper constraints.
KNN	Provides good performance but is sensitive to the choice of K and feature scaling.
Naive Bayes	Simple and fast model, performs well despite strong independence assumptions.
Random Forest	Shows excellent performance with high accuracy and MCC due to ensemble learning and reduced overfitting.
XGBoost	Delivers the best overall performance with strong generalization and high AUC, making it the most robust model in this project.
6. Streamlit Web Application

An interactive Streamlit application was developed with the following features:

CSV file upload for test data

Model selection dropdown

Display of evaluation metrics

Visualization of confusion matrix

The application loads pre-trained models and performs inference on uploaded test data without retraining.

7. Deployment

The Streamlit application was deployed using Streamlit Community Cloud and is accessible via a public URL. The deployment is connected directly to the GitHub repository and installs dependencies using the requirements.txt file.

8. How to Run the Project Locally
pip install -r requirements.txt
python -m streamlit run app.py

9. Repository Structure
ml-project/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│   ├── logisticregression.pkl
│   ├── decisiontree.pkl
│   ├── kneighbors.pkl
│   ├── gaussiannb.pkl
│   ├── randomforest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl

10. Conclusion

This project successfully demonstrates an end-to-end machine learning pipeline including model training, evaluation, deployment, and interactive visualization. Ensemble models such as Random Forest and XGBoost outperform traditional classifiers, highlighting the effectiveness of ensemble learning techniques for classification tasks.