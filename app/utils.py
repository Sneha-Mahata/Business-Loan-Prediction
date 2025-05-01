import pandas as pd
import numpy as np
import joblib

# List of expected features from the training data
EXPECTED_FEATURES = [
    'Selected', 'ApprovalDate', 'ApprovalFY', 'Term', 'NoEmp', 
    'NewExist', 'CreateJob', 'RetainedJob', 'DisbursementDate',
    'DisbursementGross', 'GrAppv', 'New', 'RealEstate', 'TermLoan', 'RevLineCr_Y'
]

def preprocess_input(input_df, scaler):
    """
    Preprocess the input data to match the format expected by the model
    
    Args:
        input_df (pd.DataFrame): Input data for prediction
        scaler (StandardScaler): Fitted scaler for numerical features
        
    Returns:
        pd.DataFrame: Preprocessed data ready for model prediction
    """
    # Make a copy to avoid modifying the original
    df = input_df.copy()
    
    # Handle missing columns by adding them with default value of 0
    for feature in EXPECTED_FEATURES:
        if feature not in df.columns:
            df[feature] = 0
    
    # Ensure column order matches the training data
    df = df[EXPECTED_FEATURES]
    
    # Apply standardization
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns
    )
    
    return df_scaled

def format_currency(amount, currency="₹"):
    """
    Format amounts in millions with currency symbol
    
    Args:
        amount (int): Amount in lowest denomination
        currency (str): Currency symbol
        
    Returns:
        str: Formatted currency string
    """
    return f"{currency}{amount/1000000:.2f}M"

def get_risk_profile(prob):
    """
    Map probability to risk category
    
    Args:
        prob (float): Default probability
        
    Returns:
        str: Risk category
    """
    if prob < 0.25:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Very High"