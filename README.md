# Car-Price-Prediction-with-Machine-Learning
This project builds a regression model to predict car prices based on various features such as mileage, horsepower, and brand.
Problem Statement

Car prices vary based on multiple factors. The goal is to create a predictive model that estimates the selling price of a car accurately using historical data.

Tools & Libraries Used
Python 🐍
Pandas
NumPy
Matplotlib / Seaborn
Scikit-learn
📁 Dataset

The dataset contains features such as:

Car Name / Brand
Year of Manufacture
Selling Price
Present Price
Kilometers Driven
Fuel Type
Transmission
Owner Type
⚙️ Project Workflow
1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
2. Load Dataset
df = pd.read_csv("car data.csv")
df.head()
3. Data Preprocessing
# Check for missing values
df.isnull().sum()

# Convert categorical variables using encoding
df = pd.get_dummies(df, drop_first=True)
4. Define Features & Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
5. Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
6. Model Training
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
7. Model Prediction
y_pred = model.predict(X_test)
8. Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
📈 Results
The model predicts car prices with reasonable accuracy.
Performance can be improved using advanced models like Random Forest or XGBoost.
🚀 How to Run This Project
# Clone repository
git clone https://github.com/AdaezeR/car-price-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run notebook or script
python app.py
📌 Future Improvements
Use advanced regression models (Random Forest, Gradient Boosting)
Perform hyperparameter tuning
Deploy model using Flask or Streamlit
👩🏽‍💻 Author

Adaeze Rose
Data Science |  Data Analyst
