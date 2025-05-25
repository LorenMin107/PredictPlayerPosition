"""
Football Player Position Prediction - Model Pipeline

This module contains functions for the complete machine learning pipeline to predict
football player positions based on performance statistics. The pipeline includes
data preprocessing, model training, evaluation, and visualization components.

The module is designed to be used both as a standalone script and as a library
imported by other scripts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pathlib
from typing import Dict, Tuple, Optional, Any

# Machine learning imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# Data Loading and Exploration Functions
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load player statistics data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing player data
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    if not pathlib.Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape} (rows, columns)")
    return df


def explore_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform initial exploration of the dataset and return key statistics.
    
    Args:
        df (pd.DataFrame): DataFrame containing player statistics
        
    Returns:
        Dict[str, Any]: Dictionary containing exploration results
    """
    # Extract primary position (first position listed)
    df['Primary_Position'] = df['Pos'].str.split(',').str[0].str.strip()
    
    # Collect exploration results
    results = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'sample_data': df.head(),
        'position_counts': df['Pos'].value_counts(),
        'primary_position_counts': df['Primary_Position'].value_counts(),
        'unique_primary_positions': df['Primary_Position'].nunique()
    }
    
    return results


# Data Preprocessing Functions
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the player data for modeling.
    
    This function:
    1. Removes rows with missing position data
    2. Extracts primary position from position string
    3. Drops identifier columns
    4. Handles missing values
    
    Args:
        df (pd.DataFrame): Raw player statistics DataFrame
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original data
    df_clean = df.copy()
    
    # 1. Remove rows where 'Pos' is NaN
    print(f"Number of rows before removing NaN Pos: {df_clean.shape[0]}")
    df_clean = df_clean[df_clean['Pos'].notna()]
    print(f"Number of rows after removing NaN Pos: {df_clean.shape[0]}")
    
    # 2. Extract primary position
    df_clean['Primary_Position'] = df_clean['Pos'].str.split(',').str[0].str.strip()
    
    # 3. Drop identifier columns
    identifier_columns = ['Rk', 'Player', 'Born']
    df_clean = df_clean.drop(columns=identifier_columns)
    
    # 4. Handle missing values
    # Identify numerical and categorical columns
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Impute numeric NaNs with median
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)
    
    # Impute categorical NaNs with "Unknown"
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna("Unknown")
    
    # Verify no missing values remain
    missing_values = df_clean.isnull().sum().sum()
    print(f"{missing_values} missing values remaining")
    
    return df_clean


# Exploratory Data Analysis Functions
def plot_position_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create a bar plot showing the distribution of player positions.
    
    Args:
        df (pd.DataFrame): DataFrame containing player data with 'Primary_Position' column
        save_path (Optional[str]): Path to save the plot image, if provided
    """
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Primary_Position', hue='Primary_Position', 
                       data=df, palette='viridis', legend=False)
    plt.title('Distribution of Player Positions', fontsize=16)
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add count labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Position distribution plot saved to {save_path}")
    
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create a heatmap showing correlations between numerical features.
    
    Args:
        df (pd.DataFrame): DataFrame containing player data
        save_path (Optional[str]): Path to save the plot image, if provided
    """
    plt.figure(figsize=(14, 12))
    
    # Select only numerical columns for correlation
    numerical_features = df.select_dtypes(include=['int64', 'float64'])
    
    # Calculate correlation matrix
    corr_matrix = numerical_features.corr()
    
    # Create heatmap with upper triangle masked
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    ax = sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                    cmap='coolwarm', linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Correlation heatmap saved to {save_path}")
    
    plt.show()


# Feature Engineering and Model Preparation Functions
def prepare_features_and_target(df: pd.DataFrame, target_column: str = 'Primary_Position') -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Separate features and target variable, and encode the target.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame
        target_column (str): Name of the target column
        
    Returns:
        Tuple[pd.DataFrame, np.ndarray, LabelEncoder]: Features DataFrame, encoded target array, and the label encoder
    """
    # Columns to exclude from features
    exclude_columns = ['Pos', target_column]
    
    # Get features and target
    X = df.drop(columns=exclude_columns)
    y = df[target_column]
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Original classes: {label_encoder.classes_}")
    print(f"Encoded classes: {np.unique(y_encoded)}")
    
    return X, y_encoded, label_encoder


def create_train_test_split(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features DataFrame
        y (np.ndarray): Target array
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create a ColumnTransformer for preprocessing numerical and categorical features.
    
    Args:
        X (pd.DataFrame): Features DataFrame
        
    Returns:
        ColumnTransformer: Preprocessor for transforming features
    """
    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical columns ({len(numerical_columns)}): {numerical_columns}")
    print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")
    
    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ],
        remainder='drop'
    )
    
    return preprocessor


# Model Training and Evaluation Functions
def get_classifiers() -> Dict[str, Any]:
    """
    Get a dictionary of classifier models to evaluate.
    
    Returns:
        Dict[str, Any]: Dictionary mapping classifier names to initialized models
    """
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=400, random_state=42),
        'XGBClassifier': XGBClassifier(
            n_estimators=600, 
            learning_rate=0.05, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42
        ),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=7)
    }


def cross_validate_models(X_train: pd.DataFrame, y_train: np.ndarray, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """
    Perform cross-validation on multiple classifiers and return performance metrics.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (np.ndarray): Training target
        preprocessor (ColumnTransformer): Feature preprocessor
        
    Returns:
        pd.DataFrame: DataFrame with cross-validation results for each classifier
    """
    # Get classifiers
    classifiers = get_classifiers()
    
    # Dictionary to store cross-validation results
    cv_results = {
        'Classifier': [],
        'Mean Accuracy': [],
        'Mean Macro-F1': []
    }
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform cross-validation for each classifier
    for name, clf in classifiers.items():
        print(f"\nCross-validating {name}...")
        
        # Create a pipeline with the preprocessor and classifier
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Perform cross-validation
        cv_scores = cross_validate(
            model_pipeline, X_train, y_train, 
            cv=cv,
            scoring=['accuracy', 'f1_macro'],
            return_train_score=False
        )
        
        # Calculate mean scores
        mean_accuracy = cv_scores['test_accuracy'].mean()
        mean_f1 = cv_scores['test_f1_macro'].mean()
        
        # Store results
        cv_results['Classifier'].append(name)
        cv_results['Mean Accuracy'].append(mean_accuracy)
        cv_results['Mean Macro-F1'].append(mean_f1)
        
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        print(f"Mean Macro-F1: {mean_f1:.4f}")
    
    # Convert to DataFrame and sort by Mean Macro-F1
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df = cv_results_df.sort_values(by='Mean Macro-F1', ascending=False)
    
    return cv_results_df


def train_best_model(X_train: pd.DataFrame, y_train: np.ndarray, preprocessor: ColumnTransformer, 
                    best_clf_name: str) -> Tuple[Pipeline, Any]:
    """
    Train the best model on the full training dataset.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (np.ndarray): Training target
        preprocessor (ColumnTransformer): Feature preprocessor
        best_clf_name (str): Name of the best classifier from cross-validation
        
    Returns:
        Tuple[Pipeline, Any]: Trained pipeline and the classifier object
    """
    # Get all classifiers
    classifiers = get_classifiers()
    
    # Get the best classifier
    best_clf = classifiers[best_clf_name]
    
    # Create the best pipeline
    best_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_clf)
    ])
    
    # Fit on full training data
    print(f"Fitting {best_clf_name} on full training data...")
    best_pipeline.fit(X_train, y_train)
    
    return best_pipeline, best_clf


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray, 
                  label_encoder: LabelEncoder, best_clf_name: str) -> Dict[str, Any]:
    """
    Evaluate the trained model on the test set and return performance metrics.
    
    Args:
        pipeline (Pipeline): Trained model pipeline
        X_test (pd.DataFrame): Test features
        y_test (np.ndarray): Test target
        label_encoder (LabelEncoder): Encoder used for the target variable
        best_clf_name (str): Name of the classifier
        
    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics
    """
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    
    # Convert encoded predictions and test labels back to original class names
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test_original, y_pred_original, output_dict=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test_original, y_pred_original)
    
    # Store evaluation results
    evaluation_results = {
        'model_name': best_clf_name,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_test_original': y_test_original,
        'y_pred_original': y_pred_original
    }
    
    return evaluation_results


def save_evaluation_results(evaluation_results: Dict[str, Any], artifacts_dir: str = 'artifacts'):
    """
    Save model evaluation results to files.
    
    Args:
        evaluation_results (Dict[str, Any]): Dictionary containing evaluation metrics
        artifacts_dir (str): Directory to save artifacts
    """
    # Create artifacts directory if it doesn't exist
    pathlib.Path(artifacts_dir).mkdir(exist_ok=True)
    
    # Extract data from evaluation results
    model_name = evaluation_results['model_name']
    accuracy = evaluation_results['accuracy']
    report = evaluation_results['classification_report']
    
    # Save classification report
    report_path = f"{artifacts_dir}/classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Classification Report for {model_name}\n\n")
        f.write(f"Accuracy on test set: {accuracy:.4f}\n\n")
        # Convert dict report back to string format
        f.write(classification_report(
            evaluation_results['y_test_original'], 
            evaluation_results['y_pred_original']
        ))
    print(f"Classification report saved to {report_path}")
    
    # Save accuracy
    accuracy_path = f"{artifacts_dir}/accuracy.txt"
    with open(accuracy_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy on test set: {accuracy:.4f}\n")
    print(f"Accuracy saved to {accuracy_path}")
    
    # Save model summary
    summary_path = f"{artifacts_dir}/model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"MODEL PERFORMANCE SUMMARY - {model_name}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy on test set: {accuracy:.4f}\n")
        f.write(f"Macro F1 Score: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted F1 Score: {report['weighted avg']['f1-score']:.4f}\n\n")
        
        f.write("Class-wise Performance:\n")
        f.write("-"*30 + "\n")
        for position in sorted(report.keys()):
            if position not in ['accuracy', 'macro avg', 'weighted avg']:
                precision = report[position]['precision']
                recall = report[position]['recall']
                f1 = report[position]['f1-score']
                support = report[position]['support']
                f.write(f"{position:15s} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Support: {support}\n")
    print(f"Performance summary saved to {summary_path}")


# Visualization Functions
def plot_confusion_matrix(evaluation_results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Create and display a confusion matrix visualization.
    
    Args:
        evaluation_results (Dict[str, Any]): Dictionary containing evaluation metrics
        save_path (Optional[str]): Path to save the plot image, if provided
    """
    # Extract data from evaluation results
    cm = evaluation_results['confusion_matrix']
    model_name = evaluation_results['model_name']
    unique_classes = np.unique(np.concatenate([
        evaluation_results['y_test_original'], 
        evaluation_results['y_pred_original']
    ]))
    
    # Create a figure with two subplots
    plt.figure(figsize=(18, 8))
    
    # 1. Raw counts confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(unique_classes), 
                yticklabels=sorted(unique_classes))
    plt.xlabel('Predicted Position', fontsize=12)
    plt.ylabel('True Position', fontsize=12)
    plt.title(f'Confusion Matrix (Counts) - {model_name}', fontsize=14)
    
    # 2. Normalized confusion matrix
    plt.subplot(1, 2, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=sorted(unique_classes), 
                yticklabels=sorted(unique_classes))
    plt.xlabel('Predicted Position', fontsize=12)
    plt.ylabel('True Position', fontsize=12)
    plt.title(f'Normalized Confusion Matrix - {model_name}', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_feature_importance(model, preprocessor: ColumnTransformer, X_train: pd.DataFrame, 
                           save_path: Optional[str] = None):
    """
    Create and display a feature importance visualization for tree-based models.
    
    Args:
        model: Trained classifier model
        preprocessor (ColumnTransformer): Feature preprocessor
        X_train (pd.DataFrame): Training features
        save_path (Optional[str]): Path to save the plot image, if provided
    """
    # Check if model is tree-based
    if not hasattr(model, 'feature_importances_') and not hasattr(model, 'get_booster'):
        print("Model is not tree-based, skipping feature importance calculation.")
        return
    
    print("Computing feature importance...")
    
    # Get the feature names after preprocessing
    # First, fit the preprocessor on the training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Get the feature names
    feature_names = []
    
    # Get numerical feature names
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for name in numerical_columns:
        feature_names.append(name)
    
    # Get one-hot encoded feature names
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    ohe = preprocessor.named_transformers_['cat']
    for i, category in enumerate(categorical_columns):
        for category_value in ohe.categories_[i]:
            feature_names.append(f"{category}_{category_value}")
    
    # Extract feature importance
    if hasattr(model, 'feature_importances_'):
        # For RandomForest
        importances = model.feature_importances_
        importance_type = "feature_importances_"
    else:
        # For XGBoost
        importance_dict = model.get_booster().get_score(importance_type='gain')
        importances = np.zeros(len(feature_names))
        for key, value in importance_dict.items():
            idx = int(key.replace('f', ''))
            if idx < len(importances):
                importances[idx] = value
        importance_type = "gain"
    
    # Create a DataFrame for easier sorting and visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Display top 15 features
    print("\nTop 15 features by importance:")
    print(feature_importance_df.head(15).to_string(index=False))
    
    # Create a bar plot of the top 15 features
    plt.figure(figsize=(12, 8))
    top_15 = feature_importance_df.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_15)
    plt.title(f'Top 15 Features by Importance ({importance_type})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


# Model Serialization Functions
def save_model(pipeline: Pipeline, model_path: str):
    """
    Save the trained model pipeline to a file.
    
    Args:
        pipeline (Pipeline): Trained model pipeline
        model_path (str): Path to save the model
    """
    print(f"Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    print(f"Model saved successfully to {model_path}")


# Main Pipeline Function
def run_full_pipeline(data_path: str, artifacts_dir: str = 'artifacts') -> Dict[str, Any]:
    """
    Run the complete machine learning pipeline from data loading to model evaluation.
    
    Args:
        data_path (str): Path to the CSV file containing player data
        artifacts_dir (str): Directory to save artifacts
        
    Returns:
        Dict[str, Any]: Dictionary containing pipeline results
    """
    # Create artifacts directory
    pathlib.Path(artifacts_dir).mkdir(exist_ok=True)
    
    # 1. Load and explore data
    df = load_data(data_path)
    exploration_results = explore_data(df)
    
    # 2. Preprocess data
    df_clean = preprocess_data(df)
    
    # 3. Prepare features and target
    X, y_encoded, label_encoder = prepare_features_and_target(df_clean)
    
    # 4. Split data
    X_train, X_test, y_train, y_test = create_train_test_split(X, y_encoded)
    
    # 5. Create preprocessor
    preprocessor = create_preprocessor(X)
    
    # 6. Cross-validate models
    cv_results = cross_validate_models(X_train, y_train, preprocessor)
    
    # 7. Get best model
    best_clf_name = cv_results.iloc[0]['Classifier']
    print(f"\nBest-performing model: {best_clf_name}")
    
    # 8. Train best model
    best_pipeline, best_clf = train_best_model(X_train, y_train, preprocessor, best_clf_name)
    
    # 9. Evaluate model
    evaluation_results = evaluate_model(best_pipeline, X_test, y_test, label_encoder, best_clf_name)
    
    # 10. Save evaluation results
    save_evaluation_results(evaluation_results, artifacts_dir)
    
    # 11. Plot confusion matrix
    plot_confusion_matrix(evaluation_results, f"{artifacts_dir}/confusion_matrix.png")
    
    # 12. Plot feature importance (if applicable)
    if best_clf_name in ['RandomForestClassifier', 'XGBClassifier']:
        plot_feature_importance(best_clf, preprocessor, X_train, f"{artifacts_dir}/feature_importance.png")
    
    # 13. Save model
    save_model(best_pipeline, f"{artifacts_dir}/position_classifier.joblib")
    
    # Return pipeline results
    return {
        'df': df,
        'df_clean': df_clean,
        'X': X,
        'y_encoded': y_encoded,
        'label_encoder': label_encoder,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'cv_results': cv_results,
        'best_clf_name': best_clf_name,
        'best_pipeline': best_pipeline,
        'best_clf': best_clf,
        'evaluation_results': evaluation_results
    }


if __name__ == "__main__":
    # Run the full pipeline when the script is executed directly
    results = run_full_pipeline('data/top5-players.csv')
    print("\nAll tasks completed successfully!")