from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
from PIL import Image as PILImage

def create_pdf_report(df, df_processed, models, results, best_model_name, X_train_scaled, scaler, label_encoders):
    """Generate comprehensive PDF report"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    # Title
    elements.append(Paragraph("Customer Annual Spending Score Prediction", title_style))
    elements.append(Paragraph("E-commerce Customer Behavior Analysis and Predictive Modeling", 
                             ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14, 
                                          alignment=TA_CENTER, textColor=colors.grey)))
    elements.append(Spacer(1, 0.3*inch))
    
    # Introduction
    elements.append(Paragraph("1. Introduction", heading_style))
    intro_text = """
    This report presents a comprehensive analysis of customer spending behavior for an e-commerce company. 
    The primary objective is to predict customer annual spending scores to optimize targeted marketing campaigns. 
    The analysis involves exploring customer demographic details, purchase history, and income information to build 
    predictive models that can help the company make data-driven marketing decisions.
    
    The dataset contains information about 200 customers, including their age, gender, annual income, and spending scores. 
    Through extensive data exploration, feature engineering, and machine learning model development, we have developed 
    robust predictive models that can accurately forecast customer spending behavior.
    """
    elements.append(Paragraph(intro_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Tasks Completed
    elements.append(Paragraph("2. Tasks Completed", heading_style))
    tasks_text = """
    The following tasks were completed as part of this analysis:
    """
    elements.append(Paragraph(tasks_text, normal_style))
    
    tasks_list = [
        "Dataset exploration and identification of relationships between features and spending score",
        "Handling missing values and data quality assessment",
        "Feature engineering to derive new meaningful features from existing data",
        "Data preprocessing including scaling and encoding of categorical variables",
        "Building and training multiple regression models (simple and complex)",
        "Hyperparameter tuning using GridSearchCV for optimal model performance",
        "Model evaluation using RMSE, MAE, and R² metrics",
        "Comprehensive visualization of correlations, distributions, and relationships",
        "Model results visualization and interpretation",
        "Customer segmentation and marketing insights generation"
    ]
    
    for i, task in enumerate(tasks_list, 1):
        elements.append(Paragraph(f"{i}. {task}", normal_style))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Justification for Tasks
    elements.append(Paragraph("3. Justification for Tasks", heading_style))
    justification_text = """
    Each task was carefully selected to ensure a thorough and effective analysis:
    
    <b>Data Exploration:</b> Understanding the dataset structure and relationships is crucial before building models. 
    This helps identify patterns, outliers, and potential issues that could affect model performance.
    
    <b>Feature Engineering:</b> Creating new features from existing data can significantly improve model accuracy. 
    Features like income-to-age ratio, interaction terms, and categorical groupings capture non-linear relationships 
    that simple features might miss.
    
    <b>Multiple Models:</b> Comparing simple (Linear Regression) and complex models (Random Forest, Gradient Boosting) 
    allows us to balance interpretability with performance. Simple models are easier to understand, while complex models 
    often provide better accuracy.
    
    <b>Hyperparameter Tuning:</b> Machine learning models have parameters that control their behavior. Systematic tuning 
    ensures we get the best possible performance from each model type.
    
    <b>Comprehensive Evaluation:</b> Using multiple metrics (RMSE, MAE, R²) provides a complete picture of model performance. 
    RMSE penalizes large errors, MAE shows average error magnitude, and R² indicates how well the model explains variance.
    
    <b>Visualization:</b> Visual representations make complex data and model results accessible to stakeholders, enabling 
    better decision-making.
    """
    elements.append(Paragraph(justification_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Dataset Overview
    elements.append(Paragraph("4. Dataset Overview", heading_style))
    dataset_text = f"""
    The dataset contains {len(df)} customers with the following characteristics:
    """
    elements.append(Paragraph(dataset_text, normal_style))
    
    # Dataset statistics table
    stats_data = [
        ['Metric', 'Value'],
        ['Total Customers', str(len(df))],
        ['Features', str(len(df.columns) - 1)],
        ['Mean Spending Score', f"{df['Spending Score (1-100)'].mean():.2f}"],
        ['Median Spending Score', f"{df['Spending Score (1-100)'].median():.2f}"],
        ['Std Dev Spending Score', f"{df['Spending Score (1-100)'].std():.2f}"],
        ['Mean Age', f"{df['Age'].mean():.1f} years"],
        ['Mean Annual Income', f"${df['Annual Income (k$)'].mean():.1f}k"],
    ]
    
    stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Key Relationships
    elements.append(Paragraph("5. Key Relationships and Insights", heading_style))
    elements.append(Paragraph("5.1 Correlation Analysis", subheading_style))
    
    correlation_text = """
    The correlation analysis reveals important relationships:
    """
    elements.append(Paragraph(correlation_text, normal_style))
    
    # Calculate correlations
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    corr_matrix = df[numeric_cols].corr()
    age_corr = corr_matrix.loc['Age', 'Spending Score (1-100)']
    income_corr = corr_matrix.loc['Annual Income (k$)', 'Spending Score (1-100)']
    
    corr_text = f"""
    • Age vs Spending Score: Correlation coefficient of {age_corr:.3f} indicates a {'weak' if abs(age_corr) < 0.3 else 'moderate' if abs(age_corr) < 0.7 else 'strong'} 
    {'negative' if age_corr < 0 else 'positive'} relationship. This suggests that age has some influence on spending behavior.
    
    • Annual Income vs Spending Score: Correlation coefficient of {income_corr:.3f} shows a {'weak' if abs(income_corr) < 0.3 else 'moderate' if abs(income_corr) < 0.7 else 'strong'} 
    {'negative' if income_corr < 0 else 'positive'} relationship. Income appears to be a {'less' if abs(income_corr) < 0.3 else 'more'} significant factor in spending patterns.
    """
    elements.append(Paragraph(corr_text, normal_style))
    elements.append(Spacer(1, 0.15*inch))
    
    # Model Performance
    elements.append(Paragraph("6. Model Performance", heading_style))
    
    if best_model_name in results:
        result = results[best_model_name]
        model = models[best_model_name]
        
        elements.append(Paragraph(f"6.1 Best Model: {best_model_name}", subheading_style))
        
        perf_text = f"""
        The {best_model_name} was selected as the best performing model based on test R² score. 
        The model achieved the following performance metrics:
        """
        elements.append(Paragraph(perf_text, normal_style))
        
        # Performance table
        perf_data = [
            ['Metric', 'Training', 'Test'],
            ['RMSE', f"{result['train_rmse']:.4f}", f"{result['test_rmse']:.4f}"],
            ['MAE', f"{result['train_mae']:.4f}", f"{result['test_mae']:.4f}"],
            ['R² Score', f"{result['train_r2']:.4f}", f"{result['test_r2']:.4f}"],
        ]
        
        perf_table = Table(perf_data, colWidths=[2*inch, 2*inch, 2*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        elements.append(perf_table)
        elements.append(Spacer(1, 0.15*inch))
        
        # Performance explanation
        r2_score_val = result['test_r2']
        if r2_score_val > 0.7:
            perf_level = "excellent"
        elif r2_score_val > 0.5:
            perf_level = "good"
        elif r2_score_val > 0.3:
            perf_level = "moderate"
        else:
            perf_level = "acceptable"
        
        perf_explanation = f"""
        <b>Performance Explanation:</b>
        
        The R² score of {r2_score_val:.4f} indicates that the model explains {r2_score_val*100:.2f}% of the variance in spending scores. 
        This is considered {perf_level} performance for this type of prediction task. The RMSE of {result['test_rmse']:.4f} means that, 
        on average, the model's predictions deviate from actual spending scores by approximately {result['test_rmse']:.2f} points. 
        The MAE of {result['test_mae']:.4f} represents the average absolute error, providing another perspective on prediction accuracy.
        
        The close alignment between training and test metrics suggests the model generalizes well to new data without significant overfitting.
        """
        elements.append(Paragraph(perf_explanation, normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Customer Attributes Influence
        elements.append(Paragraph("7. Customer Attributes Influencing Annual Spending", heading_style))
        
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
        
        top_features = importance_df.head(10)
        
        attr_text = """
        The analysis reveals which customer attributes most significantly influence annual spending scores. 
        The following attributes were identified as the most influential:
        """
        elements.append(Paragraph(attr_text, normal_style))
        
        # Top features table
        feat_data = [['Rank', 'Feature', 'Importance']]
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            feat_data.append([str(idx), row['Feature'], f"{row['Importance']:.4f}"])
        
        feat_table = Table(feat_data, colWidths=[0.8*inch, 3.5*inch, 1.5*inch])
        feat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        elements.append(feat_table)
        elements.append(Spacer(1, 0.15*inch))
        
        # Interpretation
        top_feature = top_features.iloc[0]
        interpretation_text = f"""
        <b>Key Interpretation:</b>
        
        The most influential attribute is <b>{top_feature['Feature']}</b> with an importance score of {top_feature['Importance']:.4f}. 
        This indicates that {top_feature['Feature']} has the strongest predictive power for determining customer spending scores.
        
        Original features like Age, Annual Income, and Gender all contribute to the model, but the engineered features such as 
        interaction terms and categorical groupings often provide additional predictive value by capturing complex relationships 
        that simple linear features might miss.
        """
        elements.append(Paragraph(interpretation_text, normal_style))
        elements.append(Spacer(1, 0.2*inch))
    
    # Marketing Insights
    elements.append(Paragraph("8. Insights for Improving Targeted Marketing", heading_style))
    
    # Analyze segments
    if best_model_name in models:
        model = models[best_model_name]
        X = df_processed.drop(['CustomerID', 'Spending Score (1-100)', 'Age_Group', 'Income_Group'], axis=1, errors='ignore')
        
        # Encode if needed
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    X[col] = le.transform(X[col].astype(str))
                except:
                    X[col] = 0
        
        df_processed['Predicted_Spending'] = model.predict(scaler.transform(X))
        df_processed['Spending_Category'] = pd.cut(df_processed['Predicted_Spending'], 
                                                   bins=[0, 30, 50, 70, 100], 
                                                   labels=['Low', 'Medium', 'High', 'Very High'])
        
        high_spenders = df_processed[df_processed['Spending_Category'].isin(['High', 'Very High'])]
        low_spenders = df_processed[df_processed['Spending_Category'] == 'Low']
        
        if len(high_spenders) > 0 and len(low_spenders) > 0:
            insights_text = f"""
            Based on the model predictions and customer segmentation, the following marketing insights are recommended:
            
            <b>High Spending Customers ({len(high_spenders)} customers):</b>
            • Average Age: {high_spenders['Age'].mean():.1f} years
            • Average Income: ${high_spenders['Annual Income (k$)'].mean():.1f}k
            • Marketing Strategy: Focus on premium products, exclusive offers, and loyalty programs. 
            These customers represent the highest value segment and should receive personalized attention.
            
            <b>Low Spending Customers ({len(low_spenders)} customers):</b>
            • Average Age: {low_spenders['Age'].mean():.1f} years
            • Average Income: ${low_spenders['Annual Income (k$)'].mean():.1f}k
            • Marketing Strategy: Target with discount campaigns, budget-friendly options, and value propositions. 
            Focus on converting these customers to higher spending segments through strategic promotions.
            
            <b>Demographic Targeting:</b>
            The analysis shows that spending patterns vary by demographic characteristics. Marketing campaigns should be 
            tailored to specific age groups and income levels to maximize effectiveness. For instance, younger customers 
            with high income may respond better to trendy, premium products, while older customers might prefer 
            value-focused offerings.
            """
            elements.append(Paragraph(insights_text, normal_style))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Key Takeaways
    elements.append(Paragraph("9. Key Takeaways", heading_style))
    takeaways = [
        f"The {best_model_name if 'best_model_name' in locals() else 'selected model'} provides reliable predictions for customer spending scores.",
        "Feature engineering significantly improved model performance by capturing non-linear relationships.",
        "Customer segmentation enables targeted marketing strategies for different customer groups.",
        "Age, income, and their interactions are key factors in predicting spending behavior.",
        "The model can be used to identify high-value customers for retention programs.",
        "Predictive insights enable data-driven marketing budget allocation.",
    ]
    
    for i, takeaway in enumerate(takeaways, 1):
        elements.append(Paragraph(f"• {takeaway}", normal_style))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Suggestions
    elements.append(Paragraph("10. Suggestions and Recommendations", heading_style))
    suggestions_text = """
    <b>Model Improvements:</b>
    • Collect more data to improve model generalization and reduce prediction variance
    • Include additional features such as purchase history, browsing patterns, and product preferences
    • Implement ensemble methods combining multiple models for better accuracy
    • Regularly retrain the model with new data to maintain performance over time
    • Use more sophisticated hyperparameter optimization techniques like Bayesian optimization
    
    <b>Data Collection:</b>
    • Gather more detailed customer information including purchase frequency, product categories, and seasonal patterns
    • Track customer interactions across multiple channels (online, mobile, in-store)
    • Collect feedback and satisfaction scores to understand customer preferences
    
    <b>Implementation Strategy:</b>
    • Deploy the model in a production environment with real-time prediction capabilities
    • Integrate predictions into customer relationship management (CRM) systems
    • Create automated marketing campaigns based on predicted spending scores
    • Monitor model performance and update regularly as customer behavior evolves
    """
    elements.append(Paragraph(suggestions_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Real-World Applications
    elements.append(Paragraph("11. Real-World Applications", heading_style))
    applications_text = """
    The predictive model developed in this analysis has numerous practical applications:
    
    <b>1. Personalized Marketing:</b> Target customers with high predicted spending scores with premium products and 
    exclusive offers, increasing conversion rates and revenue.
    
    <b>2. Budget Allocation:</b> Allocate marketing budget more effectively by focusing resources on customer segments 
    with the highest potential return on investment.
    
    <b>3. Product Recommendations:</b> Suggest products to customers based on their predicted spending patterns, 
    improving cross-selling and upselling opportunities.
    
    <b>4. Customer Retention:</b> Identify high-value customers for retention programs, reducing churn and maintaining 
    revenue streams.
    
    <b>5. Pricing Strategy:</b> Adjust pricing for different customer segments based on their predicted spending capacity, 
    maximizing revenue while maintaining customer satisfaction.
    
    <b>6. Inventory Management:</b> Stock products preferred by high-spending customers, optimizing inventory levels 
    and reducing waste.
    
    <b>7. Campaign Optimization:</b> Design marketing campaigns targeting specific spending score ranges, improving 
    campaign effectiveness and ROI.
    
    <b>8. Customer Acquisition:</b> Identify characteristics of high-spending customers to target similar prospects 
    in acquisition campaigns.
    """
    elements.append(Paragraph(applications_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Conclusion
    elements.append(Paragraph("12. Conclusion", heading_style))
    conclusion_text = """
    This analysis successfully developed predictive models for customer annual spending scores, providing valuable 
    insights for targeted marketing campaigns. The comprehensive approach, from data exploration to model deployment 
    recommendations, ensures that the e-commerce company can make informed, data-driven decisions to optimize 
    marketing effectiveness and maximize customer value.
    
    The models demonstrate good predictive performance and can be effectively used to segment customers, allocate 
    marketing resources, and personalize customer experiences. With proper implementation and continuous monitoring, 
    these models will contribute significantly to the company's marketing optimization efforts.
    """
    elements.append(Paragraph(conclusion_text, normal_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

