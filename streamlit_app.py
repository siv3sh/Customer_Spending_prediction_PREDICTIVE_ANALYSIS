import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style for better visualizations
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        try:
            plt.style.use('ggplot')
        except:
            pass
try:
    sns.set_palette("husl")
except:
    pass

# Set random seed for reproducibility
np.random.seed(42)

# Page configuration
st.set_page_config(
    page_title="Customer Spending Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💰 Customer Annual Spending Score Prediction</h1>', unsafe_allow_html=True)
st.markdown("### E-commerce Customer Behavior Analysis and Predictive Modeling")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Data Overview", "🔍 Exploratory Analysis", "🔧 Feature Engineering", 
     "🤖 Model Training", "📈 Model Results", "🎯 Predictions", "📋 Insights & Report"]
)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Mall_Customers.csv')
        return df
    except:
        st.error("Please ensure 'Mall_Customers.csv' is in the same directory as this app.")
        return None

@st.cache_data
def process_data(df):
    """Process and engineer features"""
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = df_processed.ffill().bfill()
    
    # Feature Engineering
    df_processed['Age_Group_Category'] = pd.cut(df_processed['Age'], 
                                                 bins=[0, 30, 40, 50, 100], 
                                                 labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    df_processed['Income_Group_Category'] = pd.cut(df_processed['Annual Income (k$)'], 
                                                    bins=[0, 40, 70, 100, 150], 
                                                    labels=['Low', 'Medium', 'High', 'Very High'])
    
    df_processed['Income_Age_Ratio'] = df_processed['Annual Income (k$)'] / (df_processed['Age'] + 1)
    df_processed['Age_Squared'] = df_processed['Age'] ** 2
    df_processed['Income_Squared'] = df_processed['Annual Income (k$)'] ** 2
    df_processed['Age_Income_Interaction'] = df_processed['Age'] * df_processed['Annual Income (k$)']
    df_processed['Spending_Capacity'] = (df_processed['Annual Income (k$)'] - df_processed['Annual Income (k$)'].min()) / \
                                         (df_processed['Annual Income (k$)'].max() - df_processed['Annual Income (k$)'].min())
    df_processed['Young_High_Income'] = ((df_processed['Age'] < 35) & (df_processed['Annual Income (k$)'] > 70)).astype(int)
    df_processed['Senior_Low_Income'] = ((df_processed['Age'] > 50) & (df_processed['Annual Income (k$)'] < 50)).astype(int)
    
    return df_processed

def process_single_customer(gender, age, annual_income, income_min, income_max):
    """Process a single customer input for prediction"""
    # Create feature dictionary
    features = {
        'Gender': gender,
        'Age': age,
        'Annual Income (k$)': annual_income
    }
    
    # Feature Engineering
    if age <= 30:
        age_group = 'Young'
    elif age <= 40:
        age_group = 'Middle'
    elif age <= 50:
        age_group = 'Senior'
    else:
        age_group = 'Elderly'
    
    if annual_income <= 40:
        income_group = 'Low'
    elif annual_income <= 70:
        income_group = 'Medium'
    elif annual_income <= 100:
        income_group = 'High'
    else:
        income_group = 'Very High'
    
    features['Age_Group_Category'] = age_group
    features['Income_Group_Category'] = income_group
    features['Income_Age_Ratio'] = annual_income / (age + 1)
    features['Age_Squared'] = age ** 2
    features['Income_Squared'] = annual_income ** 2
    features['Age_Income_Interaction'] = age * annual_income
    features['Spending_Capacity'] = (annual_income - income_min) / (income_max - income_min) if income_max > income_min else 0
    features['Young_High_Income'] = 1 if (age < 35 and annual_income > 70) else 0
    features['Senior_Low_Income'] = 1 if (age > 50 and annual_income < 50) else 0
    
    return pd.DataFrame([features])

@st.cache_resource
def train_models(X_train_scaled, y_train, X_test_scaled, y_test):
    """Train all models"""
    models = {}
    results = {}
    
    # Simple Model: Linear Regression
    simple_model = LinearRegression()
    simple_model.fit(X_train_scaled, y_train)
    models['Linear Regression'] = simple_model
    
    y_train_pred_simple = simple_model.predict(X_train_scaled)
    y_test_pred_simple = simple_model.predict(X_test_scaled)
    
    results['Linear Regression'] = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_simple)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_simple)),
        'train_mae': mean_absolute_error(y_train, y_train_pred_simple),
        'test_mae': mean_absolute_error(y_test, y_test_pred_simple),
        'train_r2': r2_score(y_train, y_train_pred_simple),
        'test_r2': r2_score(y_test, y_test_pred_simple),
        'predictions': y_test_pred_simple
    }
    
    # Complex Model 1: Random Forest
    with st.spinner("Training Random Forest (this may take a minute)..."):
        rf_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
        rf_model = RandomForestRegressor(random_state=42)
        rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='neg_mean_squared_error', 
                                      n_jobs=-1, verbose=0)
        rf_grid_search.fit(X_train_scaled, y_train)
        best_rf_model = rf_grid_search.best_estimator_
        models['Random Forest'] = best_rf_model
        
        y_train_pred_rf = best_rf_model.predict(X_train_scaled)
        y_test_pred_rf = best_rf_model.predict(X_test_scaled)
        
        results['Random Forest'] = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_rf)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_rf)),
            'train_mae': mean_absolute_error(y_train, y_train_pred_rf),
            'test_mae': mean_absolute_error(y_test, y_test_pred_rf),
            'train_r2': r2_score(y_train, y_train_pred_rf),
            'test_r2': r2_score(y_test, y_test_pred_rf),
            'predictions': y_test_pred_rf,
            'best_params': rf_grid_search.best_params_
        }
    
    # Complex Model 2: Gradient Boosting
    with st.spinner("Training Gradient Boosting (this may take a minute)..."):
        gb_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2]
        }
        gb_model = GradientBoostingRegressor(random_state=42)
        gb_grid_search = GridSearchCV(gb_model, gb_param_grid, cv=3, scoring='neg_mean_squared_error', 
                                      n_jobs=-1, verbose=0)
        gb_grid_search.fit(X_train_scaled, y_train)
        best_gb_model = gb_grid_search.best_estimator_
        models['Gradient Boosting'] = best_gb_model
        
        y_train_pred_gb = best_gb_model.predict(X_train_scaled)
        y_test_pred_gb = best_gb_model.predict(X_test_scaled)
        
        results['Gradient Boosting'] = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_gb)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_gb)),
            'train_mae': mean_absolute_error(y_train, y_train_pred_gb),
            'test_mae': mean_absolute_error(y_test, y_test_pred_gb),
            'train_r2': r2_score(y_train, y_train_pred_gb),
            'test_r2': r2_score(y_test, y_test_pred_gb),
            'predictions': y_test_pred_gb,
            'best_params': gb_grid_search.best_params_
        }
    
    return models, results

# Load data
df = load_data()

if df is not None:
    df_processed = process_data(df)
    
    # Prepare features
    X = df_processed.drop(['CustomerID', 'Spending Score (1-100)', 'Age_Group', 'Income_Group'], axis=1, errors='ignore')
    y = df_processed['Spending Score (1-100)']
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # PAGE 1: Data Overview
    if page == "📊 Data Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Data Completeness", f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types:**")
            st.write(df.dtypes)
        with col2:
            st.write("**Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum() / len(df) * 100).values
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0] if missing_df['Missing Count'].sum() > 0 else pd.DataFrame({'Message': ['No missing values!']}))
        
        st.subheader("Target Variable Distribution")
        
        # Interactive Plotly chart
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Distribution of Spending Score', 'Box Plot of Spending Score'))
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df['Spending Score (1-100)'], nbinsx=20, name='Frequency', marker_color='skyblue'),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=df['Spending Score (1-100)'], name='Spending Score', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, title_text="Spending Score Analysis")
        fig.update_xaxes(title_text="Spending Score", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Spending Score", row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{df['Spending Score (1-100)'].mean():.2f}")
        with col2:
            st.metric("Median", f"{df['Spending Score (1-100)'].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{df['Spending Score (1-100)'].std():.2f}")
    
    # PAGE 2: Exploratory Analysis
    elif page == "🔍 Exploratory Analysis":
        st.header("🔍 Exploratory Data Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Correlation Analysis", "Age vs Spending", "Income vs Spending", "Gender Analysis", "3D Visualization"]
        )
        
        if analysis_type == "Correlation Analysis":
            st.subheader("Correlation Matrix")
            numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            correlation_matrix = df[numeric_cols].corr()
            
            # Interactive Plotly heatmap
            fig = px.imshow(correlation_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           color_continuous_scale='RdBu',
                           title='Correlation Matrix - Features vs Spending Score',
                           labels=dict(color="Correlation"))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Correlation with Spending Score:**")
            st.dataframe(correlation_matrix['Spending Score (1-100)'].sort_values(ascending=False))
        
        elif analysis_type == "Age vs Spending":
            st.subheader("Age vs Spending Score Analysis")
            
            # Interactive scatter plot
            fig1 = px.scatter(df, x='Age', y='Spending Score (1-100)', 
                             title='Age vs Spending Score',
                             labels={'Age': 'Age', 'Spending Score (1-100)': 'Spending Score'},
                             hover_data=['Gender', 'Annual Income (k$)'],
                             color='Spending Score (1-100)',
                             color_continuous_scale='Viridis')
            fig1.update_traces(marker=dict(size=8, opacity=0.6))
            st.plotly_chart(fig1, use_container_width=True)
            
            # Regression plot
            fig2 = px.scatter(df, x='Age', y='Spending Score (1-100)', 
                             trendline="ols",
                             title='Age vs Spending Score (with Regression Line)',
                             labels={'Age': 'Age', 'Spending Score (1-100)': 'Spending Score'})
            st.plotly_chart(fig2, use_container_width=True)
            
            # Box plot by age groups
            age_bins = pd.cut(df['Age'], bins=5)
            df_temp = df.copy()
            df_temp['Age_Group'] = age_bins.astype(str)
            fig3 = px.box(df_temp, x='Age_Group', y='Spending Score (1-100)',
                          title='Spending Score by Age Groups',
                          labels={'Age_Group': 'Age Group', 'Spending Score (1-100)': 'Spending Score'})
            fig3.update_xaxes(tickangle=45)
            st.plotly_chart(fig3, use_container_width=True)
        
        elif analysis_type == "Income vs Spending":
            st.subheader("Annual Income vs Spending Score Analysis")
            
            # Interactive scatter plot
            fig1 = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                             title='Annual Income vs Spending Score',
                             labels={'Annual Income (k$)': 'Annual Income (k$)', 'Spending Score (1-100)': 'Spending Score'},
                             hover_data=['Gender', 'Age'],
                             color='Spending Score (1-100)',
                             color_continuous_scale='Greens')
            fig1.update_traces(marker=dict(size=8, opacity=0.6))
            st.plotly_chart(fig1, use_container_width=True)
            
            # Regression plot
            fig2 = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                             trendline="ols",
                             title='Annual Income vs Spending Score (with Regression Line)',
                             labels={'Annual Income (k$)': 'Annual Income (k$)', 'Spending Score (1-100)': 'Spending Score'})
            st.plotly_chart(fig2, use_container_width=True)
            
            # Box plot by income groups
            income_bins = pd.cut(df['Annual Income (k$)'], bins=5)
            df_temp = df.copy()
            df_temp['Income_Group'] = income_bins.astype(str)
            fig3 = px.box(df_temp, x='Income_Group', y='Spending Score (1-100)',
                          title='Spending Score by Income Groups',
                          labels={'Income_Group': 'Income Group', 'Spending Score (1-100)': 'Spending Score'})
            fig3.update_xaxes(tickangle=45)
            st.plotly_chart(fig3, use_container_width=True)
        
        elif analysis_type == "Gender Analysis":
            st.subheader("Gender Analysis")
            
            # Pie chart
            gender_counts = df['Gender'].value_counts()
            fig1 = px.pie(values=gender_counts.values, names=gender_counts.index,
                         title='Gender Distribution')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Box plot
            fig2 = px.box(df, x='Gender', y='Spending Score (1-100)',
                         title='Spending Score by Gender',
                         labels={'Gender': 'Gender', 'Spending Score (1-100)': 'Spending Score'})
            st.plotly_chart(fig2, use_container_width=True)
            
            # Violin plot
            fig3 = px.violin(df, x='Gender', y='Spending Score (1-100)',
                            title='Spending Score Distribution by Gender',
                            labels={'Gender': 'Gender', 'Spending Score (1-100)': 'Spending Score'})
            st.plotly_chart(fig3, use_container_width=True)
            
            st.write("**Spending Score Statistics by Gender:**")
            st.dataframe(df.groupby('Gender')['Spending Score (1-100)'].describe())
        
        elif analysis_type == "3D Visualization":
            st.subheader("3D Relationship: Age, Income, and Spending Score")
            
            # Interactive 3D scatter plot
            fig = px.scatter_3d(df, 
                              x='Age', 
                              y='Annual Income (k$)', 
                              z='Spending Score (1-100)',
                              color='Spending Score (1-100)',
                              color_continuous_scale='Viridis',
                              hover_data=['Gender'],
                              title='3D Relationship: Age, Income, and Spending Score',
                              labels={'Age': 'Age', 
                                     'Annual Income (k$)': 'Annual Income (k$)', 
                                     'Spending Score (1-100)': 'Spending Score'})
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 3: Feature Engineering
    elif page == "🔧 Feature Engineering":
        st.header("🔧 Feature Engineering")
        
        st.subheader("Engineered Features")
        st.write(f"**Original Features:** {len(df.columns)}")
        st.write(f"**Engineered Features:** {len(df_processed.columns)}")
        st.write(f"**Total Features for Modeling:** {len(X.columns)}")
        
        st.subheader("New Features Created")
        feature_list = [
            "Age_Group_Category: Categorical age groups (Young, Middle, Senior, Elderly)",
            "Income_Group_Category: Categorical income groups (Low, Medium, High, Very High)",
            "Income_Age_Ratio: Income to Age ratio (spending power indicator)",
            "Age_Squared: Age squared for non-linear relationships",
            "Income_Squared: Income squared for non-linear relationships",
            "Age_Income_Interaction: Interaction between Age and Income",
            "Spending_Capacity: Normalized spending capacity",
            "Young_High_Income: Binary flag for young customers with high income",
            "Senior_Low_Income: Binary flag for senior customers with low income"
        ]
        
        for feature in feature_list:
            st.write(f"• {feature}")
        
        st.subheader("Sample of Engineered Features")
        feature_cols = ['Age', 'Annual Income (k$)', 'Age_Group_Category', 'Income_Group_Category', 
                        'Income_Age_Ratio', 'Age_Squared', 'Income_Squared', 'Age_Income_Interaction',
                        'Spending_Capacity', 'Young_High_Income', 'Senior_Low_Income']
        st.dataframe(df_processed[feature_cols].head(10), use_container_width=True)
        
        st.subheader("Feature Statistics")
        st.dataframe(df_processed[feature_cols].describe(), use_container_width=True)
    
    # PAGE 4: Model Training
    elif page == "🤖 Model Training":
        st.header("🤖 Model Training")
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                models, results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
                st.session_state['models'] = models
                st.session_state['results'] = results
                st.session_state['scaler'] = scaler
                st.session_state['label_encoders'] = label_encoders
                st.session_state['X_train_scaled'] = X_train_scaled
                st.success("✅ Models trained successfully!")
        
        if 'results' in st.session_state:
            st.subheader("Model Performance Comparison")
            
            comparison_data = {
                'Model': [],
                'Train RMSE': [],
                'Test RMSE': [],
                'Train MAE': [],
                'Test MAE': [],
                'Train R²': [],
                'Test R²': []
            }
            
            for model_name, result in st.session_state['results'].items():
                comparison_data['Model'].append(model_name)
                comparison_data['Train RMSE'].append(result['train_rmse'])
                comparison_data['Test RMSE'].append(result['test_rmse'])
                comparison_data['Train MAE'].append(result['train_mae'])
                comparison_data['Test MAE'].append(result['test_mae'])
                comparison_data['Train R²'].append(result['train_r2'])
                comparison_data['Test R²'].append(result['test_r2'])
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Find best model
            best_model_idx = comparison_df['Test R²'].idxmax()
            best_model_name = comparison_df.loc[best_model_idx, 'Model']
            st.session_state['best_model'] = best_model_name
            
            st.success(f"🏆 **Best Model: {best_model_name}** (R² = {comparison_df.loc[best_model_idx, 'Test R²']:.4f})")
            
            # Interactive visualization
            fig = make_subplots(rows=1, cols=3, 
                              subplot_titles=('Test RMSE Comparison', 'Test MAE Comparison', 'Test R² Comparison'))
            
            colors_list = ['skyblue', 'lightgreen', 'coral']
            for i, col in enumerate(['Test RMSE', 'Test MAE', 'Test R²']):
                fig.add_trace(
                    go.Bar(x=comparison_df['Model'], y=comparison_df[col], 
                          name=col, marker_color=colors_list[i]),
                    row=1, col=i+1
                )
            
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(tickangle=45, row=1, col=1)
            fig.update_xaxes(tickangle=45, row=1, col=2)
            fig.update_xaxes(tickangle=45, row=1, col=3)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show hyperparameters for complex models
            st.subheader("Best Hyperparameters")
            for model_name, result in st.session_state['results'].items():
                if 'best_params' in result:
                    st.write(f"**{model_name}:**")
                    st.json(result['best_params'])
        else:
            st.info("👆 Click the button above to train the models.")
    
    # PAGE 5: Model Results
    elif page == "📈 Model Results":
        st.header("📈 Model Results & Visualizations")
        
        if 'results' not in st.session_state:
            st.warning("⚠️ Please train the models first in the 'Model Training' page.")
        else:
            model_selection = st.selectbox(
                "Select Model to View",
                list(st.session_state['results'].keys())
            )
            
            result = st.session_state['results'][model_selection]
            model = st.session_state['models'][model_selection]
            predictions = result['predictions']
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test RMSE", f"{result['test_rmse']:.4f}")
            with col2:
                st.metric("Test MAE", f"{result['test_mae']:.4f}")
            with col3:
                st.metric("Test R²", f"{result['test_r2']:.4f}")
            
            # Interactive Visualizations
            st.subheader("Model Performance Visualizations")
            
            # Actual vs Predicted
            fig1 = px.scatter(x=y_test, y=predictions,
                            title=f'Actual vs Predicted - {model_selection}',
                            labels={'x': 'Actual Spending Score', 'y': 'Predicted Spending Score'},
                            hover_data=[y_test.index])
            # Add perfect prediction line
            min_val = min(y_test.min(), predictions.min())
            max_val = max(y_test.max(), predictions.max())
            fig1.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                    mode='lines', name='Perfect Prediction',
                                    line=dict(color='red', dash='dash', width=2)))
            fig1.update_traces(marker=dict(size=6, opacity=0.6, color='steelblue'))
            st.plotly_chart(fig1, use_container_width=True)
            
            # Residuals
            residuals = y_test - predictions
            fig2 = px.scatter(x=predictions, y=residuals,
                            title='Residual Plot',
                            labels={'x': 'Predicted Spending Score', 'y': 'Residuals'})
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            fig2.update_traces(marker=dict(size=6, opacity=0.6, color='coral'))
            st.plotly_chart(fig2, use_container_width=True)
            
            # Distribution of Residuals
            fig3 = px.histogram(x=residuals, nbins=20,
                              title='Distribution of Residuals',
                              labels={'x': 'Residuals', 'y': 'Frequency'})
            fig3.add_vline(x=0, line_dash="dash", line_color="red")
            fig3.update_traces(marker_color='lightgreen', opacity=0.7)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Actual vs Predicted Line Plot
            test_indices = list(range(len(y_test)))
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=test_indices, y=y_test.values, mode='lines+markers',
                                     name='Actual', line=dict(color='steelblue')))
            fig4.add_trace(go.Scatter(x=test_indices, y=predictions, mode='lines+markers',
                                     name='Predicted', line=dict(color='coral')))
            fig4.update_layout(title='Actual vs Predicted Over Test Samples',
                             xaxis_title='Test Sample Index',
                             yaxis_title='Spending Score',
                             height=400)
            st.plotly_chart(fig4, use_container_width=True)
            
            # Feature Importance
            st.subheader("Feature Importance")
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': X_train_scaled.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance.head(15), 
                           x='Importance', y='Feature',
                           orientation='h',
                           title=f'Top 15 Feature Importance - {model_selection}',
                           labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
                           color='Importance',
                           color_continuous_scale='Viridis')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(feature_importance.head(15), use_container_width=True)
            else:
                feature_importance = pd.DataFrame({
                    'Feature': X_train_scaled.columns,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                fig = px.bar(feature_importance.head(15),
                           x='Coefficient', y='Feature',
                           orientation='h',
                           title=f'Top 15 Feature Coefficients - {model_selection}',
                           labels={'Coefficient': 'Coefficient Value', 'Feature': 'Feature'},
                           color='Coefficient',
                           color_continuous_scale='RdBu',
                           color_continuous_midpoint=0)
                fig.add_vline(x=0, line_dash="dash", line_color="black")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(feature_importance.head(15), use_container_width=True)
    
    # PAGE 6: Predictions
    elif page == "🎯 Predictions":
        st.header("🎯 Make Predictions")
        
        if 'models' not in st.session_state:
            st.warning("⚠️ Please train the models first in the 'Model Training' page.")
        else:
            st.subheader("Predict Spending Score for New Customer")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", 18, 80, 35)
                annual_income = st.slider("Annual Income (k$)", 15, 140, 60)
            
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female"])
                model_selection = st.selectbox("Select Model", list(st.session_state['models'].keys()))
            
            if st.button("🔮 Predict Spending Score", type="primary"):
                # Get income min/max from training data for normalization
                income_min = df['Annual Income (k$)'].min()
                income_max = df['Annual Income (k$)'].max()
                
                # Process single customer
                input_processed = process_single_customer(gender, age, annual_income, income_min, income_max)
                
                # Prepare features - ensure same columns as training data
                X_input = input_processed.copy()
                
                # Encode categorical variables
                for col in X_input.select_dtypes(include=['object', 'category']).columns:
                    if col in st.session_state['label_encoders']:
                        le = st.session_state['label_encoders'][col]
                        try:
                            X_input[col] = le.transform(X_input[col].astype(str))
                        except:
                            # If new category, use first encoded value
                            X_input[col] = 0
                
                # Ensure all columns from training are present
                X_train_cols = st.session_state['X_train_scaled'].columns
                for col in X_train_cols:
                    if col not in X_input.columns:
                        X_input[col] = 0
                
                # Reorder columns to match training data
                X_input = X_input[X_train_cols]
                
                # Scale
                X_input_scaled = st.session_state['scaler'].transform(X_input)
                X_input_scaled = pd.DataFrame(X_input_scaled, columns=X_input.columns)
                
                # Predict
                model = st.session_state['models'][model_selection]
                prediction = model.predict(X_input_scaled)[0]
                
                st.success(f"### Predicted Spending Score: **{prediction:.2f}**")
                
                # Show prediction category
                if prediction < 30:
                    category = "Low Spender"
                    color = "red"
                elif prediction < 50:
                    category = "Medium Spender"
                    color = "orange"
                elif prediction < 70:
                    category = "High Spender"
                    color = "blue"
                else:
                    category = "VIP Spender"
                    color = "green"
                
                st.markdown(f"**Customer Category:** <span style='color:{color}'>{category}</span>", unsafe_allow_html=True)
            
            st.subheader("Batch Predictions")
            uploaded_file = st.file_uploader("Upload CSV file with customer data", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write("**Uploaded Data:**")
                    st.dataframe(batch_df.head(), use_container_width=True)
                    
                    if st.button("🔮 Predict for All Customers", type="primary"):
                        # Check required columns
                        required_cols = ['Gender', 'Age', 'Annual Income (k$)']
                        if not all(col in batch_df.columns for col in required_cols):
                            st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                        else:
                            # Process batch
                            batch_processed = process_data(batch_df)
                            X_batch = batch_processed.drop(['CustomerID', 'Spending Score (1-100)', 'Age_Group', 'Income_Group'], axis=1, errors='ignore')
                            
                            # Encode
                            for col in X_batch.select_dtypes(include=['object', 'category']).columns:
                                if col in st.session_state['label_encoders']:
                                    le = st.session_state['label_encoders'][col]
                                    try:
                                        X_batch[col] = le.transform(X_batch[col].astype(str))
                                    except:
                                        X_batch[col] = 0
                            
                            # Ensure all columns from training are present
                            X_train_cols = st.session_state['X_train_scaled'].columns
                            for col in X_train_cols:
                                if col not in X_batch.columns:
                                    X_batch[col] = 0
                            
                            # Reorder columns to match training data
                            X_batch = X_batch[X_train_cols]
                            
                            # Scale
                            X_batch_scaled = st.session_state['scaler'].transform(X_batch)
                            X_batch_scaled = pd.DataFrame(X_batch_scaled, columns=X_batch.columns)
                            
                            # Predict
                            model = st.session_state['models'][model_selection]
                            predictions = model.predict(X_batch_scaled)
                        
                        # Create results dataframe
                        results_df = batch_df.copy()
                        results_df['Predicted_Spending_Score'] = predictions
                        results_df['Customer_Segment'] = pd.cut(predictions, 
                                                               bins=[0, 30, 50, 70, 100], 
                                                               labels=['Low Spender', 'Medium Spender', 'High Spender', 'VIP'])
                        
                        st.write("**Predictions:**")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv,
                            file_name="customer_predictions.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    # PAGE 7: Insights & Report
    elif page == "📋 Insights & Report":
        st.header("📋 Insights & Reporting")
        
        if 'results' not in st.session_state:
            st.warning("⚠️ Please train the models first in the 'Model Training' page.")
        else:
            best_model_name = st.session_state.get('best_model', 'Linear Regression')
            result = st.session_state['results'][best_model_name]
            model = st.session_state['models'][best_model_name]
            
            st.subheader("1. Model Performance Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Model", best_model_name)
            with col2:
                st.metric("Test R²", f"{result['test_r2']:.4f}")
            with col3:
                st.metric("Test RMSE", f"{result['test_rmse']:.4f}")
            with col4:
                st.metric("Test MAE", f"{result['test_mae']:.4f}")
            
            r2_score_val = result['test_r2']
            if r2_score_val > 0.7:
                performance_level = "Excellent"
            elif r2_score_val > 0.5:
                performance_level = "Good"
            elif r2_score_val > 0.3:
                performance_level = "Moderate"
            else:
                performance_level = "Poor"
            
            st.info(f"**Performance Level:** {performance_level} - The model explains {r2_score_val*100:.2f}% of the variance in spending scores.")
            
            st.subheader("2. Customer Attributes Influencing Annual Spending")
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': X_train_scaled.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
            else:
                importance_df = pd.DataFrame({
                    'Feature': X_train_scaled.columns,
                    'Importance': np.abs(model.coef_)
                }).sort_values('Importance', ascending=False)
            
            st.write("**Top 10 Most Influential Attributes:**")
            st.dataframe(importance_df.head(10), use_container_width=True)
            
            st.subheader("3. Marketing Insights & Recommendations")
            
            # Analyze segments - prepare X properly
            X = df_processed.drop(['CustomerID', 'Spending Score (1-100)', 'Age_Group', 'Income_Group'], axis=1, errors='ignore')
            
            # Encode categorical variables to match training data
            X_encoded = X.copy()
            for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
                if col in label_encoders:
                    X_encoded[col] = label_encoders[col].transform(X_encoded[col].astype(str))
            
            # Ensure columns match training data
            X_train_cols = X_train_scaled.columns
            X_encoded = X_encoded[X_train_cols]
            
            # Scale and predict
            X_scaled = scaler.transform(X_encoded)
            df_processed['Predicted_Spending'] = model.predict(X_scaled)
            df_processed['Spending_Category'] = pd.cut(df_processed['Predicted_Spending'], 
                                                       bins=[0, 30, 50, 70, 100], 
                                                       labels=['Low', 'Medium', 'High', 'Very High'])
            
            segment_analysis = df_processed.groupby('Spending_Category').agg({
                'Age': 'mean',
                'Annual Income (k$)': 'mean',
                'Gender': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
            }).round(2)
            
            st.write("**Customer Segments by Predicted Spending:**")
            st.dataframe(segment_analysis, use_container_width=True)
            
            st.write("**Marketing Recommendations:**")
            
            high_spenders = df_processed[df_processed['Spending_Category'].isin(['High', 'Very High'])]
            if len(high_spenders) > 0:
                st.success(f"""
                **High Spending Customers:**
                - Average Age: {high_spenders['Age'].mean():.1f} years
                - Average Income: ${high_spenders['Annual Income (k$)'].mean():.1f}k
                - Focus: Premium products, loyalty programs, exclusive offers
                """)
            
            low_spenders = df_processed[df_processed['Spending_Category'] == 'Low']
            if len(low_spenders) > 0:
                st.info(f"""
                **Low Spending Customers:**
                - Average Age: {low_spenders['Age'].mean():.1f} years
                - Average Income: ${low_spenders['Annual Income (k$)'].mean():.1f}k
                - Focus: Discount campaigns, budget-friendly options, value propositions
                """)
            
            st.subheader("4. Suggested Improvements")
            improvements = [
                "Collect More Data: Increase dataset size for better generalization",
                "Additional Features: Include purchase history, browsing patterns, product preferences",
                "Feature Engineering: Create more domain-specific features (e.g., customer lifetime value)",
                "Ensemble Methods: Combine multiple models for better predictions",
                "Regular Updates: Retrain model periodically with new data",
                "Cross-Validation: Use k-fold cross-validation for more robust evaluation",
                "Hyperparameter Optimization: Use more sophisticated methods (Bayesian Optimization)",
                "Feature Selection: Remove less important features to reduce overfitting"
            ]
            
            for i, improvement in enumerate(improvements, 1):
                st.write(f"{i}. {improvement}")
            
            st.subheader("5. Real-World Applications")
            applications = [
                "Personalized Marketing: Target customers with high predicted spending scores",
                "Budget Allocation: Allocate marketing budget based on customer segments",
                "Product Recommendations: Suggest products to customers based on spending patterns",
                "Customer Retention: Identify high-value customers for retention programs",
                "Pricing Strategy: Adjust pricing for different customer segments",
                "Inventory Management: Stock products preferred by high-spending customers",
                "Campaign Optimization: Design campaigns targeting specific spending score ranges",
                "Customer Acquisition: Identify characteristics of high-spending customers for targeting"
            ]
            
            for i, application in enumerate(applications, 1):
                st.write(f"{i}. {application}")
            
            # Customer Segmentation Visualization
            st.subheader("6. Customer Segmentation Analysis")
            if 'models' in st.session_state:
                model = st.session_state['models'][st.session_state.get('best_model', 'Linear Regression')]
                
                # Prepare X for prediction (reuse the same X_encoded from above)
                X_seg = df_processed.drop(['CustomerID', 'Spending Score (1-100)', 'Age_Group', 'Income_Group', 'Predicted_Spending'], axis=1, errors='ignore')
                
                # Encode categorical variables to match training data
                X_seg_encoded = X_seg.copy()
                for col in X_seg_encoded.select_dtypes(include=['object', 'category']).columns:
                    if col in label_encoders:
                        X_seg_encoded[col] = label_encoders[col].transform(X_seg_encoded[col].astype(str))
                
                # Ensure columns match training data
                X_train_cols = X_train_scaled.columns
                X_seg_encoded = X_seg_encoded[X_train_cols]
                
                # Scale and predict
                X_seg_scaled = scaler.transform(X_seg_encoded)
                df_processed['Customer_Segment_Pred'] = model.predict(X_seg_scaled)
                df_processed['Customer_Segment'] = pd.cut(df_processed['Customer_Segment_Pred'], 
                                                       bins=[0, 30, 50, 70, 100], 
                                                       labels=['Low Spender', 'Medium Spender', 'High Spender', 'VIP'])
                
                # Pie chart
                segment_counts = df_processed['Customer_Segment'].value_counts()
                fig1 = px.pie(values=segment_counts.values, names=segment_counts.index,
                             title='Customer Segment Distribution')
                st.plotly_chart(fig1, use_container_width=True)
                
                # Box plots
                col1, col2 = st.columns(2)
                with col1:
                    fig2 = px.box(df_processed, x='Customer_Segment', y='Age',
                                 title='Age Distribution by Customer Segment')
                    fig2.update_xaxes(tickangle=45)
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    fig3 = px.box(df_processed, x='Customer_Segment', y='Annual Income (k$)',
                                 title='Income Distribution by Customer Segment')
                    fig3.update_xaxes(tickangle=45)
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Gender distribution
                segment_gender = pd.crosstab(df_processed['Customer_Segment'], df_processed['Gender'])
                fig4 = px.bar(segment_gender, barmode='group',
                             title='Gender Distribution by Customer Segment',
                             labels={'value': 'Count', 'Customer_Segment': 'Customer Segment'})
                fig4.update_xaxes(tickangle=45)
                st.plotly_chart(fig4, use_container_width=True)
                
                st.write("**Customer Segment Summary:**")
                st.dataframe(segment_counts.to_frame('Count'), use_container_width=True)

if __name__ == "__main__":
    pass

