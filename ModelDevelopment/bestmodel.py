# Main Function to Get the Best Model Across Experiments

import mlflow
import os
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from Validation.XGBoost import XGBoostPM25Model
from Validation.Prophet import ProphetPM25Model
from Validation.RandomForest import RandomForestPM25Model
from mlflow.pyfunc import PythonModel

class PM25ModelWrapper(PythonModel):
    def __init__(self, model):
        self.model = model  # Your pre-trained model

    def predict(self, context, model_input):
        # Adjust based on how your model expects input and outputs predictions
        return self.model.predict(model_input)

# Model Loading Functions
def load_randomforest_model(filepath):
    """Loads the Random Forest model from the given filepath."""
    randomforest = RandomForestPM25Model(train_file=None, test_file=None, lambda_value=None, model_save_path=filepath)
    model_save_path = filepath
    with open(model_save_path, 'rb') as f:
       randomforest = pd.read_pickle(f)
    return randomforest

def load_xgboost_model(filepath):
    """Loads the XGBoost model from the given filepath."""
    xgbmodel = xgb.XGBRegressor(random_state=42)
    xgbmodel.load_model(filepath)
    return xgbmodel

def load_prophet_model(filepath):
    """Loads the Prophet model from the given filepath."""
    prophetmodel = ProphetPM25Model(train_file=None, test_file=None, lambda_value=None, model_save_path=filepath)
    with open(filepath, 'rb') as f:
            prophetmodel= pickle.load(f)
    return prophetmodel

# Function to Identify Best Model
def find_best_model_run(experiment_name):
    print(experiment_name)
    """Finds the best model run in an experiment based on the lowest RMSE metric."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None, None, None
    
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    if runs.empty:
        print(f"No runs found for experiment '{experiment_name}'.")
        return None, None, None
    
    # Identify the run with the lowest RMSE
    best_run = runs.loc[runs['metrics.RMSE'].idxmin()]
    best_rmse = best_run['metrics.RMSE']
    best_run_id = best_run['run_id']
    
    return best_run_id, best_rmse, experiment_name

def get_bias_results(experiment_names):
    slicing_features = ['hour', 'day_of_week', 'month', 'season']
    best_metrics = {}

    for experiment_name in experiment_names:
        # Get the experiment ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            continue
        experiment_id = experiment.experiment_id

        # Query all runs in the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment_id])

        # Ensure there are runs to process
        if runs.empty:
            print(f"No runs found for experiment '{experiment_name}'.")
            continue

        # Initialize a dictionary for this experiment to store best metric values
        best_metrics[experiment_name] = {}

        for feature in slicing_features:
            # Find the best runs based on the specific metrics for each feature
            best_mae_run = runs.loc[runs[f'metrics.{feature}_{experiment_name}_Avg_MAE'].idxmin()]
            best_rmse_run = runs.loc[runs[f'metrics.{feature}_{experiment_name}_Avg_RMSE'].idxmin()]
            best_r2_run = runs.loc[runs[f'metrics.{feature}_{experiment_name}_Avg_R2'].idxmax()]
            best_mbe_run = runs.loc[runs[f'metrics.{feature}_{experiment_name}_Avg_MBE'].idxmin()]

            # Store only the metric values and run IDs for the best runs
            best_metrics[experiment_name][f"MAE_{feature}"] = {
                "value": best_mae_run[f"metrics.{feature}_{experiment_name}_Avg_MAE"],
                "run_id": best_mae_run["run_id"]
            }
            best_metrics[experiment_name][f"RMSE_{feature}"] = {
                "value": best_rmse_run[f"metrics.{feature}_{experiment_name}_Avg_RMSE"],
                "run_id": best_rmse_run["run_id"]
            }
            best_metrics[experiment_name][f"R2_{feature}"] = {
                "value": best_r2_run[f"metrics.{feature}_{experiment_name}_Avg_R2"],
                "run_id": best_r2_run["run_id"]
            }
            best_metrics[experiment_name][f"MBE_{feature}"] = {
                "value": best_mbe_run[f"metrics.{feature}_{experiment_name}_Avg_MBE"],
                "run_id": best_mbe_run["run_id"]
            }

    return best_metrics

# Main Function to Get the Best Model Across Experiments
def get_best_model_and_load_weights(experiment_names):
    """Finds and loads the best model across multiple experiments based on RMSE."""
    best_rmse = float('inf')
    best_model = None
    best_experiment_name = None
    best_run_id = None
    rmse_results = {}
    
    for experiment_name in experiment_names:
        run_id, rmse, exp_name = find_best_model_run(experiment_name)
        if experiment_name == "PM2.5 Prophet":
            rmse_results["Prophet"] = rmse
        if experiment_name == "PM2.5 Random Forest":
            rmse_results["Random"] = rmse
        if experiment_name == "PM2.5 XGBoost Prediction":
            rmse_results["XGBoost"] = rmse

        if run_id is not None and rmse < best_rmse:
            best_rmse = rmse
            best_experiment_name = exp_name
            best_run_id = run_id

    if best_run_id:
        #model_uri, model_loader = get_model_uri_and_loader(best_experiment_name, best_run_id)
        if best_experiment_name == "PM2.5 Prophet":
            curr_dir = os.getcwd()
            directory_weights_path = os.path.join(curr_dir,"weights/prophet_pm25_model.pth")
            best_model = load_prophet_model(directory_weights_path)
        if best_experiment_name == "PM2.5 Random Forest":
            curr_dir = os.getcwd()
            directory_weights_path = os.path.join(curr_dir,"weights/randomforest_pm25_model.pth")
            best_model = load_randomforest_model(directory_weights_path)
        if best_experiment_name == "PM2.5 XGBoost Prediction":
            curr_dir = os.getcwd()
            directory_weights_path = os.path.join(curr_dir,"weights/xgboost_pm25_model.pth")
            best_model = load_xgboost_model(directory_weights_path)

        print(f"Best model found in experiment '{best_experiment_name}' with run ID '{best_run_id}'")
        print(f"Validation RMSE: {best_rmse}")
    else:
        print("No valid models found across experiments.")
    
    return best_model, best_rmse, best_experiment_name, best_run_id, rmse_results

# Define a penalty for bias metrics (lower values preferred)
def compute_bias_score(bias_metrics, metric_weights):
    score = 0
    for feature, metrics in bias_metrics.items():
        for metric, details in metrics.items():
            metric_name = metric.split('_')[0]  # Extract MAE, RMSE, etc.
            if metric_name in metric_weights:
                score += metric_weights[metric_name] * abs(details["value"])
    return score

# Main function to select the best model based on RMSE and bias scores
def select_best_model(rmse_results, bias_results, metric_weights):
    best_model = None
    best_combined_score = float('inf')
    
    for model, rmse in rmse_results.items():
        # Skip models without RMSE values
        if rmse is None:
            continue
        
        # Calculate bias score if bias results are available
        bias_key = f"{model} Bias Evaluation"
        if bias_key in bias_results:
            bias_score = compute_bias_score(bias_results[bias_key], metric_weights)
        else:
            bias_score = float('inf')  # Penalize models without bias metrics

        # Compute combined score as weighted RMSE + Bias Score
        combined_score = metric_weights["RMSE"] * rmse + bias_score
        print(f"{model} - RMSE: {rmse}, Bias Score: {bias_score}, Combined Score: {combined_score}")
        
        # Update best model if the combined score is lower
        if combined_score < best_combined_score:
            best_model = model
            best_combined_score = combined_score

    print(f"\nBest model selected: {best_model} with combined score: {best_combined_score}")
    return best_model, best_combined_score


def check_existing_model_rmse(model_name, current_rmse):
    """Check if an existing model in MLflow registry has a better or equal RMSE."""
    client = mlflow.tracking.MlflowClient()
    try:
        latest_versions = client.get_latest_versions(model_name, stages=["None", "Production", "Staging"])
        best_rmse = float('inf')
        
        for version in latest_versions:
            run_id = version.run_id
            metric_history = client.get_metric_history(run_id, 'RMSE')
            if metric_history:
                rmse = metric_history[-1].value  # Fetch the latest RMSE logged for this version
                if rmse < best_rmse:
                    best_rmse = rmse
        return best_rmse if best_rmse != float('inf') else None
    except mlflow.exceptions.RestException as e:
        print(f"Model '{model_name}' not found in registry. Assuming no previous model exists.")
        return None  # No model registered yet

def main():
    experiment_names = ["PM2.5 Random Forest", "PM2.5 XGBoost Prediction", "PM2.5 Prophet"]
    experiment_names_2 = ["Random Forest Bias Evaluation", "XGBoost Bias Evaluation", "Prophet Bias Evaluation"]
    
    # Load the best model and retrieve RMSE results and bias results
    model, best_rmse, best_experiment_name, best_run_id, rmse_results = get_best_model_and_load_weights(experiment_names)
    bias_results = get_bias_results(experiment_names_2)
    wrapped_model = PM25ModelWrapper(model)
    
    # Define weights for each metric - adjust these based on importance
    metric_weights = {
        "RMSE": 0.5,        # Overall importance of RMSE
        "MAE": 0.2,         # Mean Absolute Error weight
        "R2": 0.2,          # R-squared weight
        "MBE": 0.1          # Mean Bias Error weight
    }
    # rollback mechanishm
    # Run the selection function
    best_model_name, best_combined_score = select_best_model(rmse_results, bias_results, metric_weights)
    
    # Determine model path and load model based on the best model name
    curr_dir = os.getcwd()
    if best_model_name == "Prophet":
        directory_weights_path = os.path.join(curr_dir, "weights/prophet_pm25_model.pth")
        model = load_prophet_model(directory_weights_path)
        model_name = "PM2.5_Prophet_Model"
    elif best_model_name == "Random":
        directory_weights_path = os.path.join(curr_dir, "weights/randomforest_pm25_model.pth")
        model = load_randomforest_model(directory_weights_path)
        model_name = "PM2.5_RandomForest_Model"
    elif best_model_name == "XGBoost":
        directory_weights_path = os.path.join(curr_dir, "weights/xgboost_pm25_model.pth")
        model = load_xgboost_model(directory_weights_path)
        model_name = "PM2.5_XGBoost_Model"
    else:
        print("No valid model found.")
        return

    # Check existing model RMSE in registry for rollback
    existing_rmse = check_existing_model_rmse(model_name, best_rmse)
    if existing_rmse is not None and existing_rmse <= best_rmse:
        print(f"Existing model in registry has a better or equal RMSE ({existing_rmse}). Skipping registration.")
    else:
        # Log and register the new best model in MLflow
        with mlflow.start_run(run_id=best_run_id) as run:
            mlflow.log_param("best_combined_score", best_combined_score)
            
            # Log and push the best model to the MLflow registry
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=wrapped_model,
                registered_model_name=model_name
            )
            print(f"Best model '{model_name}' registered with MLflow Model Registry.")

    print(f"\nBest model selected: {best_model_name} with combined score: {best_combined_score}")
    print("Model Details:", best_model_name, best_combined_score)
    print("Bias Results:", bias_results)
    print("RMSE Results:", rmse_results)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
