# Predicting Football Player Positions Using Performance Statistics from Top 5 European Leagues

This project uses machine learning to predict football player positions based on performance statistics from the top 5
European leagues. It demonstrates how to use player statistics like expected goals (xG), assists, progressive passes,
and defensive actions to classify players into their primary positions.

## Overview

The project analyzes a comprehensive dataset of player statistics from top European leagues to:

- Predict player positions using various performance metrics
- Identify key statistics that influence position classification
- Provide insights into the relationship between player statistics and their roles

## Dataset

The dataset used is curated by Orkun Aktas and contains season-level statistics for thousands of players from the top 5
European leagues, including:

- Player demographics (age, nationality)
- Performance metrics (goals, assists, passes)
- Advanced statistics (xG, progressive passes, defensive actions)

## Project Structure

```
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── football_position_prediction.ipynb           # Main analysis notebook
├── artifacts/             # Model outputs and visualizations
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── position_classifier.joblib
├── data/                  # Dataset directory
│   └── top5-players.csv   
└── models/               # Model storage directory
```

## Setup and Installation

1. Clone the repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main analysis is contained in `football_position_prediction.ipynb`. To run the analysis:

1. Ensure you have Jupyter installed
2. Open the notebook:
   ```bash
   jupyter notebook football_position_prediction.ipynb
   ```
3. Run all cells to:
    - Load and preprocess the data
    - Perform exploratory data analysis
    - Train and evaluate models
    - Generate visualizations

## Model Performance

The project compares several classification models:

- Logistic Regression
- Random Forest
- XGBoost
- K-Nearest Neighbors

Model evaluation results and feature importance analysis can be found in the `artifacts` directory:

- `classification_report.txt`: Detailed model performance metrics
- `confusion_matrix.png`: Visualization of prediction accuracy
- `feature_importance.png`: Most influential features for position prediction

## Using the Trained Model

The best-performing model is saved as `position_classifier.joblib` in the artifacts directory. You can load and use it
for predictions:

```python
import joblib

# Load the model
model = joblib.load('artifacts/position_classifier.joblib')

# Make predictions (ensure your input data matches the training features)
predictions = model.predict(X_new)
```

## Requirements

Main dependencies include:

- Python 3.x
- scikit-learn
- pandas
- numpy
- xgboost
- matplotlib
- seaborn

See `requirements.txt` for complete list of dependencies.
