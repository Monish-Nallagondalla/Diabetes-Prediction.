# Diabetes Prediction

This project aims to predict whether a person has diabetes based on various health metrics using machine learning algorithms. The system uses a trained model to classify individuals as either diabetic or non-diabetic, leveraging data such as glucose levels, blood pressure, age, BMI, and more.

---

## **Project Overview**

Diabetes is a major health concern worldwide. This project builds a prediction model using historical medical data to predict whether a person has diabetes or not. The data consists of attributes like glucose levels, BMI, and age, and the prediction is based on a supervised learning approach.

---

## **Repository Contents**

- **application.py**: Main Flask application to run the prediction web service.
- **Dataset**: Contains the dataset used for training and testing the model.
- **Model**: 
  - `modelForPrediction.pkl`: A trained machine learning model for making predictions.
  - `standardScalar.pkl`: A saved StandardScaler object to scale input data for predictions.
- **Notebook**: 
  - `Decision_tree_SVC.ipynb`: Jupyter notebook implementing Decision Tree and SVC models.
  - `Logistic_Regression.ipynb`: Jupyter notebook implementing the Logistic Regression model.
- **Templates**: 
  - `home.html`: Main page for the web application.
  - `index.html`: Index page of the web app.
  - `single_prediction.html`: Page displaying individual predictions.
- **data.json**: Input data used for testing or predictions in JSON format.
- **requirements.txt**: Lists the required dependencies to run the project.
- **LICENSE.md**: License file for the project.
- **README.md**: Project documentation.

---

## **Dataset Columns**

The dataset consists of the following columns:

- **Pregnancies**: Number of pregnancies the patient has had.
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body Mass Index (weight in kg / (height in m)^2).
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
- **Age**: Age of the patient.
- **Outcome**: 0 or 1, indicating if the person has diabetes (1) or not (0).

---

## **Technologies Used**

- **Programming Language**: Python
- **Libraries**:
  - `Flask`: Web framework for deploying the model.
  - `scikit-learn`: Machine learning library for building models.
  - `pandas`, `numpy`: Data manipulation libraries.
  - `matplotlib`, `seaborn`: Visualization libraries.
- **Environment**: Jupyter Notebook for model development and Flask for web service.

---

## **How to Run the Project**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Monish-Nallagondalla/Diabetes-Prediction.git
   cd Diabetes-Prediction
2. **Install Dependencies**:

  ```bash
  pip install -r requirements.txt
```
3. **Run the Flask Application**:

  ```bash
  python application.pyt
  ```

4. **Access the Web Application**: Open your web browser and go to http://127.0.0.1:5000/ to start predicting diabetes outcomes.
   
---

**Model Details**
The project uses the following machine learning models:

Decision Tree Classifier: Used in Decision_tree_SVC.ipynb for classifying diabetes outcomes.

Support Vector Classifier (SVC): Implemented in SVC complete for better accuracy in predictions.

Logistic Regression: Implemented in Logistic_Regression.ipynb for classification tasks.

The trained models and their associated scaler are saved in the Model folder as modelForPrediction.pkl and standardScalar.pkl.

---

**Project Workflow**
Data Preprocessing: Data is cleaned, missing values are handled, and features are scaled using StandardScaler.

Model Training: Different models (Decision Tree, SVC, Logistic Regression) are trained on the dataset.

Prediction: The trained model is used to predict diabetes outcomes based on new input data.

Web Application: The Flask web app allows users to input their data and get predictions in real-time.

---

**Future Work**
Model Improvement: Experiment with more advanced algorithms to improve prediction accuracy.

Deployment: Deploy the Flask application to a cloud platform for public access.

Feature Expansion: Add more features (such as family medical history) to enhance prediction reliability.

---

**License**
This project is licensed under the MIT License. See the LICENSE.md file for more details.

**Contact**
For any questions or collaborations, feel free to reach out to me:

Name: Monish Nallagondalla
GitHub: Monish-Nallagondalla
