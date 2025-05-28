"""
Football Player Position Prediction Tool

This script demonstrates how to use the trained machine learning model to predict
a football player's position based on their performance statistics.

The model was trained on data from the top 5 European leagues and can classify players
into four positions: Defender (DF), Forward (FW), Goalkeeper (GK), and Midfielder (MF).
"""

import pandas as pd
import joblib
import os
from typing import Dict, Any


def load_model(model_path: str):
    """
    Load a trained machine learning model from the specified path.

    Args:
        model_path (str): Path to the saved model file

    Returns:
        The loaded model object

    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"Loading model from {model_path}...")
    return joblib.load(model_path)


def create_player_data(player_stats: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a DataFrame with player statistics in the format expected by the model.

    Args:
        player_stats (Dict[str, Any]): Dictionary containing player statistics

    Returns:
        pd.DataFrame: DataFrame containing the player data in the correct format
    """
    return pd.DataFrame(player_stats)


def predict_position(model, player_data: pd.DataFrame) -> int:
    """
    Predict a player's position using the trained model.

    Args:
        model: Trained machine learning model
        player_data (pd.DataFrame): Player statistics in DataFrame format

    Returns:
        int: Predicted position code (0: DF, 1: FW, 2: GK, 3: MF)
    """
    return model.predict(player_data)[0]


def get_position_mapping() -> Dict[int, str]:
    """
    Get the mapping between position codes and position names.

    Returns:
        Dict[int, str]: Dictionary mapping position codes to position names
    """
    return {
        0: "DF",  # Defender
        1: "FW",  # Forward
        2: "GK",  # Goalkeeper
        3: "MF"   # Midfielder
    }


def display_results(predicted_code: int, actual_position: str):
    """
    Display the prediction results in a user-friendly format.

    Args:
        predicted_code (int): The predicted position code
        actual_position (str): The actual position of the player (if known)
    """
    position_mapping = get_position_mapping()

    print("\nEncoding target variable:")
    for code, position in position_mapping.items():
        print(f"{position}: {code}", end=", " if code < 3 else "\n")

    print(f"\nPredicted position: {predicted_code} ({position_mapping.get(predicted_code, 'Unknown')})")
    print(f"Actual position: {actual_position}")


def main():
    """
    Main function to demonstrate the player position prediction pipeline.

    This function:
    1. Loads the trained model
    2. Creates sample player data (Bruno Fernandes)
    3. Makes a position prediction
    4. Displays the results
    """
    # Load the trained model
    model_path = 'artifacts/position_classifier.joblib'
    model = load_model(model_path)

    # Create sample player data using Bruno Fernandes' statistics
    bruno_stats = {
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
    }
    player_data = create_player_data(bruno_stats)

    # Make prediction
    predicted_position = predict_position(model, player_data)

    # Display results
    display_results(predicted_position, "MF, FW")


# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()
