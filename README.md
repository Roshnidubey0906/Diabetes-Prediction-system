# 🩺 Diabetes Prediction System

A Machine Learning web application that predicts whether a person may have diabetes based on selected health-related input features. The application uses a Support Vector Machine (SVM) classifier and provides an interactive dashboard built with Streamlit.

## 🚀 Live Demo

**Try the application:** https://diabetes-prediction-system-rosh.streamlit.app/

## 📌 Project Overview

This project demonstrates how Machine Learning can be integrated into a web application. Users can enter health-related values and receive a model prediction through an interactive interface.

This is an educational project and is **not intended for medical diagnosis or treatment decisions**.

## ✨ Features

- Interactive Streamlit dashboard
- Diabetes prediction using an SVM classifier
- Input fields for eight health-related features
- Data scaling before prediction
- Prediction history during the current app session
- CSV download for prediction history
- Dashboard metrics and model evaluation visualizations

## 🛠️ Technologies Used

- **Python** — application logic
- **Streamlit** — web interface
- **Scikit-learn** — SVM model and feature scaling
- **Pandas** — tabular data handling
- **NumPy** — numerical operations
- **Matplotlib** — visualizations
- **Joblib** — loading the trained model and scaler

## 📊 Model Information

- **Algorithm:** Support Vector Machine (SVM)
- **Kernel:** Linear
- **Test accuracy:** 77%

The model was evaluated using a test dataset. Accuracy alone does not describe all aspects of model performance. The diabetic-class recall was approximately 52%, which means the model missed a substantial proportion of diabetic cases in the test set.

## 📥 Input Features

The model accepts the following eight features:

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Roshnidubey0906/Diabetes-Prediction-system.git
cd Diabetes-Prediction-system
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the application

```bash
python -m streamlit run app.py
```

The application will display a local URL in the terminal. Open that URL in your browser.

## 📁 Project Structure

```text
Diabetes-Prediction-system/
├── app.py
├── requirements.txt
├── Diabetes_Prediction.ipynb
├── diabetes.csv
├── diabetes_scaler.pkl
└── diabetes_svm_model.pkl
```

## 🔮 Future Improvements

- Compare SVM with Logistic Regression, Random Forest and other suitable algorithms.
- Improve model evaluation and diabetic-class recall.
- Add explainable AI visualizations.
- Improve input validation and accessibility.
- Develop a more robust machine learning pipeline.

## ⚠️ Disclaimer

This application is for educational and demonstration purposes only. Its predictions are not a medical diagnosis. Do not use this application to make healthcare decisions.

## 👩‍💻 Author

**Roshni Dubey**

B.Tech Computer Science and Engineering | AI & Machine Learning

GitHub: https://github.com/Roshnidubey0906