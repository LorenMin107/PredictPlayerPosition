import pandas as pd
import joblib

# Load the trained pipeline
model = joblib.load('artifacts/position_classifier.joblib')

# Create sample player data using Bruno Fernandes' statistics
player_data = pd.DataFrame({
    'Age': [28],
    'MP': [35],
    'Starts': [35],
    'Min': [3118],
    '90s': [34.6],
    'Gls': [10],
    'Ast': [8],
    'G+A': [18],
    'G-PK': [6],
    'PK': [4],
    'PKatt': [5],
    'CrdY': [9],
    'CrdR': [0],
    'xG': [10.0],
    'npxG': [6.1],
    'xAG': [11.8],
    'npxG+xAG': [17.8],
    'PrgC': [86],
    'PrgP': [297],
    'PrgR': [182],
    'Gls_90': [0.29],
    'Ast_90': [0.23],
    'G+A_90': [0.52],
    'G-PK_90': [0.17],
    'G+A-PK_90': [0.40],
    'xG_90': [0.29],
    'xAG_90': [0.34],
    'xG+xAG_90': [0.63],
    'npxG_90': [0.18],
    'npxG+xAG_90': [0.51],
    'Squad': ['Manchester Utd'],
    'Comp': ['Premier League'],
    'Nation': ['Portugal']
})

# Make prediction
predicted_position = model.predict(player_data)

# Output mapping and result
print("Encoding target variable:")
print("DF: 0, FW: 1, GK: 2, MF: 3\n")
print(f"Predicted position: {predicted_position[0]}")
print("Actual position of Bruno Fernandes: MF, FW")
