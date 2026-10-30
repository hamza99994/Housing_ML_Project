# Housing Price Prediction and Classification

## Project Overview

This project is a machine learning web application built using Python, FastAPI, HTML, CSS, and JavaScript.

It allows users to select different machine learning models, enter house details, generate predictions, and compare results from multiple models.

## Dataset

- **Dataset:** Housing.csv
- **Records:** 545
- **Columns:** 13
- **Regression Target:** House Price
- **Classification Target:** Furnishing Status

## Machine Learning Models

**Regression Models:**
1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Support Vector Regression (SVR)

**Classification Models:**
1. Logistic Regression
2. Decision Tree (Gini)
3. Decision Tree (Entropy)
4. Random Forest Classification
5. Support Vector Machine (SVM)
6. Naive Bayes

## Data Preprocessing

The preprocessing steps include:

- Handling missing values
- Encoding categorical variables
- Scaling numerical features where required
- Splitting data into training and testing sets

The trained models are saved using Joblib.

## Web Application

The application uses:

- **FastAPI:** Backend and prediction API
- **HTML:** Website structure
- **CSS:** Website styling
- **JavaScript:** Form handling and API requests
- **Joblib:** Loading saved machine learning models

Users can select an individual model or compare multiple regression or classification models.

## How to Run

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Start the FastAPI server**

```bash
python -m uvicorn app:app --reload
```

**3. Open the website**

```text
http://127.0.0.1:8000/
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Project Files

- `Data_Processing.ipynb` — Data preprocessing
- Other `.ipynb` files — Model training and evaluation
- `saved_models/` — Trained machine learning models
- `app.py` — FastAPI backend
- `static/html/index.html` — Frontend and JavaScript
- `static/css/style.css` — Website styling
- `requirements.txt` — Python dependencies

## Conclusion

This project demonstrates the implementation and comparison of multiple machine learning algorithms through a web-based application.