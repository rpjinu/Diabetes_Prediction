# Diabetes_Prediction
"An interactive Streamlit app that predicts diabetes using user-provided medical data and a pre-trained machine learning model."

<img src="https://github.com/rpjinu/Diabetes_Prediction/blob/main/diabetes_prediction_image.png" width=800>

# Diabetes Prediction Project (End-to-End with API Using Streamlit)

This project builds a machine learning model to predict the likelihood of diabetes based on various health metrics. It covers the entire machine learning pipeline, from data preprocessing and model training to deployment as an interactive web application and API using Streamlit.

## Project Overview

The goal is to create a user-friendly application that allows individuals to input their health data and receive a prediction regarding their risk of developing diabetes. This project demonstrates a complete machine learning workflow, including:

*   **Data Preprocessing:** Handling missing values, feature scaling, and data splitting.
*   **Model Selection and Training:** Training and evaluating various machine learning models.
*   **Model Optimization:** Tuning hyperparameters and using cross-validation.
*   **Streamlit Application:** Creating an interactive web interface for predictions.
*   **API with Streamlit:** Exposing the model as an API.
*   **Deployment:** Deploying the application for public access.

## Dataset Overview

The dataset used in this project contains the following features:

*   **Pregnancies:** Number of pregnancies.
*   **Glucose:** Plasma glucose concentration in an oral glucose tolerance test (OGTT).
*   **BloodPressure:** Diastolic blood pressure (mm Hg).
*   **SkinThickness:** Triceps skin fold thickness (mm).
*   **Insulin:** Insulin level in the blood (mu U/ml).
*   **BMI:** Body Mass Index (weight (kg) / height (m)^2).
*   **DiabetesPedigreeFunction:** Diabetes pedigree function (family history score).
*   **Age:** Age of the individual.
*   **Outcome:** Target variable (1 = Diabetes, 0 = No Diabetes).

## End-to-End Steps

1.  **Data Preprocessing:**
    *   Loading the dataset using pandas.
    *   Handling missing values (e.g., imputation with mean/median).
    *   Feature scaling (e.g., MinMaxScaler or StandardScaler).
    *   Splitting data into training and testing sets.
    *   Encoding the target variable.

2.  **Model Selection and Training:**
    *   Choosing a suitable algorithm (e.g., Logistic Regression, Decision Trees, Random Forest, SVM).
    *   Training the model on the training data.
    *   Evaluating the model using metrics like accuracy, precision, recall, F1-score, and confusion matrices.

3.  **Model Optimization:**
    *   Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    *   K-fold cross-validation.

4.  **Streamlit Application:**
    *   Creating a user interface with input fields for health data.
    *   Loading the trained model using joblib or pickle.
    *   Displaying predictions based on user input.

5.  **API with Streamlit:**
    *   Creating a prediction function that preprocesses input and uses the trained model.
    *   Running the Streamlit app as a local server.

6.  **Deployment:**
    *   Deploying the app to a cloud platform like Heroku or Streamlit Sharing.

## Technologies Used

*   Python
*   Pandas
*   Scikit-learn
*   Streamlit
*   Joblib/Pickle (for model saving)
*   (Optional) Heroku/Streamlit Sharing (for deployment)

## How to Run

1.  Clone the repository: `git clone <repository_url>`
2.  Install required packages: `pip install -r requirements.txt`
3.  Run the Streamlit app: `streamlit run app.py` (replace `app.py` with your main Streamlit file)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
