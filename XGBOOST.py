import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Function to load and concatenate multiple CSV files
def load_pll_data(data_folder):
    all_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.csv')]
    return all_files

# Signal-to-Noise Ratio (SNR) calculation
def calculate_snr(signal, noise):
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noise))
    return 10 * np.log10(signal_power / noise_power)

# Main function
def train_xgboost_model(train_folder, test_folder, input_features, target_feature, random_state=42):
    # Load the training datasets
    print("Loading training data...")
    train_files = load_pll_data(train_folder)
    train_data_frames = [pd.read_csv(file) for file in train_files]
    train_data = pd.concat(train_data_frames, ignore_index=True)

    # Load the testing datasets
    print("Loading testing data...")
    test_files = load_pll_data(test_folder)

    # Train and evaluate the model for each test file
    for test_file in test_files:
        test_data = pd.read_csv(test_file)

        # Split data into input (X) and output (y)
        X_train = train_data[input_features]
        y_train = train_data[target_feature]
        X_test = test_data[input_features]
        y_test = test_data[target_feature]

        # Extract simulation time for plotting
        sim_time = test_data["Sim_time"] if "Sim_time" in test_data.columns else np.arange(len(y_test))

        # Train the XGBoost model
        print(f"Training XGBoost model for {os.path.basename(test_file)}...")
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=random_state,
            max_depth=6,               # Tree depth
            learning_rate=0.1,         # Learning rate
            n_estimators=200,          # Number of trees
            subsample=0.8,             # Subsampling
            colsample_bytree=0.8,      # Feature sampling
            reg_alpha=0.1,             # L1 regularization
            reg_lambda=1.0,            # L2 regularization
            min_child_weight=1         # Minimum child weight
        )
        model.fit(X_train, y_train)

        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_pred, y_test)
        snr = calculate_snr(y_test, y_test - y_pred)

        print(f"Model Evaluation for {os.path.basename(test_file)}:\nMean Absolute Error (MAE): {mae:.4f}\nMean Squared Error (MSE): {mse:.4f}\nRoot Mean Squared Error (RMSE): {rmse:.4f}\nR^2 Score: {r2:.4f}\nSignal-to-Noise Ratio (SNR): {snr:.4f}")

        # Plot actual vs predicted values
        print("Plotting predictions...")
        plt.figure(figsize=(12, 6))
        plt.plot(sim_time, y_test.values, label='Actual Signal', color='blue', linewidth=1)
        plt.plot(sim_time, y_pred, label='Predicted Signal', color='red', linewidth=1, alpha=0.7)
        plt.xlabel("Simulation Time")
        plt.ylabel(target_feature)
        plt.title(f"Actual vs Predicted Signal for {target_feature}")
        plt.legend()

        # Add metrics as text on the plot (positioned outside the graph)
        metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nSNR: {snr:.4f} dB"
        plt.text(1.02, 0.5, metrics_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

        # Adjust layout to make space for the text
        plt.tight_layout()

        # Create folders for saving outputs in the current working directory
        cwd = os.getcwd()
        output_folders = [os.path.join(cwd, "xgboost/graphs"), os.path.join(cwd, "xgboost/metrics"), os.path.join(cwd, "xgboost/models"), os.path.join(cwd, "xgboost/predictions")]
        for folder in output_folders:
            os.makedirs(folder, exist_ok=True)

        # Save the graph with the name of the CSV file
        graph_filename = os.path.splitext(os.path.basename(test_file))[0] + "_actual_vs_predicted.png"
        graph_path = os.path.join(cwd, f"xgboost/graphs/{graph_filename}")
        plt.savefig(graph_path, bbox_inches='tight')
        print(f"Graph saved at {graph_path}")
        plt.close()

        # Save metrics
        metrics = pd.DataFrame({
            "Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R^2 Score", "Signal-to-Noise Ratio (SNR)"],
            "Value": [mae, mse, rmse, r2, snr]
        })
        metrics_filename = os.path.splitext(os.path.basename(test_file))[0] + "_metrics.csv"
        metrics_path = os.path.join(cwd, f"xgboost/metrics/{metrics_filename}")
        metrics.to_csv(metrics_path, index=False)
        print(f"Metrics saved at {metrics_path}")

        # Save the model
        model_filename = os.path.splitext(os.path.basename(test_file))[0] + "_xgboost_model.json"
        model_path = os.path.join(cwd, f"xgboost/models/{model_filename}")
        model.save_model(model_path)
        print(f"Model saved at {model_path}")

        # Save predictions
        predictions = pd.DataFrame({"Simulation Time": sim_time, "Actual": y_test, "Predicted": y_pred})
        predictions_filename = os.path.splitext(os.path.basename(test_file))[0] + "_predictions.csv"
        predictions_path = os.path.join(cwd, f"xgboost/predictions/{predictions_filename}")
        predictions.to_csv(predictions_path, index=False)
        print(f"Predictions saved at {predictions_path}")

if __name__ == "__main__":
    # Current working directory
    cwd = os.getcwd()

    # Paths to the folders containing train and test datasets
    train_folder = os.path.join(cwd, "train")
    test_folder = os.path.join(cwd, "test")

    # Load input and output features from files
    input_signals_path = os.path.join(cwd, "input_signals.txt")
    output_signals_path = os.path.join(cwd, "output_signals.txt")

    with open(input_signals_path, "r") as f:
        input_features = f.read().splitlines()
    with open(output_signals_path, "r") as f:
        target_feature = f.read().strip()

    # Train the model
    train_xgboost_model(train_folder, test_folder, input_features, target_feature)