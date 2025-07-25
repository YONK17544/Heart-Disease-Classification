# 🫀 Heart Disease Classification Project

## 📌 Overview

This project focuses on building a machine learning model to predict whether a patient has heart disease based on clinical features. The dataset consists of **303 patient records** provided in an **Excel file**. Using data preprocessing, model training, evaluation, and hyperparameter tuning techniques, the goal is to identify the most accurate and reliable classifier for this healthcare application.

---

## 🧰 Technologies Used

- **Programming Language:** Python
- **Libraries & Tools:**
  - `NumPy`, `Pandas` – data manipulation
  - `Matplotlib`, `Seaborn` – data visualization
  - `scikit-learn` – machine learning models, training, evaluation, and tuning

---

## 📊 Dataset

- **Format:** Microsoft Excel (`.xlsx`)
- **Records:** 303 patients
- **Features:** Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG results, max heart rate, exercise-induced angina, oldpeak, slope, number of vessels colored, thalassemia, etc.
- **Target Variable:** Presence of heart disease (`1`) or absence (`0`)

---

## 📁 Project Structure

heart-disease-classification/
│
├── data/
│ └── heart_disease_data.xlsx # Excel dataset (303 records)
│
├── notebooks/
│ └── heart_disease_classification.ipynb
│
├── images/ # Visualizations and plots
├── models/ # Saved model files (if any)
├── README.md # Project overview and documentation
└── requirements.txt # Python dependencies


---

## 🚀 Models Implemented

Three classification models were evaluated:

| Model                   | Accuracy           |
|------------------------|--------------------|
| Logistic Regression    | **0.8852**          |
| K-Nearest Neighbors    | 0.6885             |
| Random Forest Classifier | 0.8361           |

After hyperparameter tuning, **Logistic Regression** was selected as the best-performing model.

---

## 🔧 Hyperparameter Tuning

`RandomizedSearchCV` and `GridSearchCV` were used to fine-tune model parameters. The best parameters for Logistic Regression were:

```python
clf = LogisticRegression(
    C=0.20433597178569418, 
    solver="liblinear"
)

Cross-validation Accuracy: 0.8447

📈 Evaluation Metrics
To evaluate model performance, the following metrics were used:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

ROC Curve

These metrics ensured balanced performance and robustness on unseen data.

✅ Key Takeaways
Logistic Regression outperformed more complex models, suggesting the dataset may be linearly separable.

KNN was the least effective, likely due to sensitivity to scale or irrelevant features.

Cross-validation ensured that results were not dependent on a specific train-test split.

Feature importance can be used for domain insights and further optimization.


📦 Requirements
See requirements.txt for all dependencies.

📚 Acknowledgments
Dataset based on 303 patients from an Excel file

Inspired by the UCI Heart Disease dataset

Built with Python and scikit-learn
