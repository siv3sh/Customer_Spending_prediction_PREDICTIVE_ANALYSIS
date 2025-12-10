# Customer Annual Spending Score Prediction

A comprehensive machine learning application for predicting customer annual spending scores to optimize targeted marketing campaigns.

## Features

- 📊 **Data Overview**: Complete dataset exploration and statistics
- 🔍 **Exploratory Analysis**: Interactive visualizations including correlation analysis, age/income relationships, gender analysis, and 3D visualizations
- 🔧 **Feature Engineering**: Automatic creation of 9+ engineered features
- 🤖 **Model Training**: Three regression models with hyperparameter tuning:
  - Simple Model: Linear Regression
  - Complex Models: Random Forest & Gradient Boosting
- 📈 **Model Results**: Comprehensive performance metrics (RMSE, MAE, R²) and visualizations
- 🎯 **Predictions**: Interactive prediction interface for single customers and batch predictions
- 📋 **Insights & Report**: Detailed analysis, marketing recommendations, and customer segmentation

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Running the Jupyter Notebook

Open `customer_spending_prediction.ipynb` in Jupyter Notebook or JupyterLab and run all cells.

## Project Structure

```
PA_ETE/
├── Mall_Customers.csv                    # Dataset
├── customer_spending_prediction.ipynb    # Jupyter notebook with full analysis
├── streamlit_app.py                      # Streamlit web application
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## Dataset

The dataset (`Mall_Customers.csv`) contains:
- CustomerID: Unique identifier
- Gender: Male/Female
- Age: Customer age
- Annual Income (k$): Annual income in thousands
- Spending Score (1-100): Target variable to predict

## Model Performance

The application trains and compares three models:
1. **Linear Regression** (Simple Model)
2. **Random Forest Regressor** (Complex Model with hyperparameter tuning)
3. **Gradient Boosting Regressor** (Complex Model with hyperparameter tuning)

Models are evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

## Features

### Engineered Features
- Age groups (Young, Middle, Senior, Elderly)
- Income groups (Low, Medium, High, Very High)
- Income-to-Age ratio
- Polynomial features (Age², Income²)
- Interaction features (Age × Income)
- Spending capacity (normalized)
- Binary flags for specific customer segments

## Streamlit App Pages

1. **📊 Data Overview**: Dataset statistics and basic information
2. **🔍 Exploratory Analysis**: Interactive visualizations and correlation analysis
3. **🔧 Feature Engineering**: View all engineered features
4. **🤖 Model Training**: Train and compare models
5. **📈 Model Results**: Detailed model performance and visualizations
6. **🎯 Predictions**: Make predictions for new customers
7. **📋 Insights & Report**: Marketing insights and recommendations

## Output Files

- `customer_spending_predictions.csv`: Contains predictions for all customers with segmentation

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

## License

This project is for educational purposes.

