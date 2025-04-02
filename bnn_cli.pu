import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

#STEP 1------------------------------------------------------------------------------------------------------
# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load the data
def load_data(filepath):
    """Load CSV data and prepare it for the BNN model"""
    df = pd.read_csv(filepath)
    
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for required columns
    required_columns = ['Symbol', 'BERT_Embeddings', 'Sentiment_Polarity', 'Sentiment_Subjectivity']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure 'Change (%)' is properly calculated if not already
    if 'Change (%)' not in df.columns:
        if 'Previous Close' in df.columns and 'New Close' in df.columns:
            df['Change (%)'] = ((df['New Close'] - df['Previous Close']) / df['Previous Close']) * 100
            print("Calculated 'Change (%)' from 'Previous Close' and 'New Close'")
        else:
            raise ValueError("Cannot calculate 'Change (%)': missing 'Previous Close' or 'New Close'")
    
    # Check for missing values in important columns
    for col in required_columns + ['Change (%)']:
        missing = df[col].isna().sum()
        print(f"Missing values in {col}: {missing} ({missing/len(df):.2%})")
        
        # Handle missing values if necessary
        if col == 'BERT_Embeddings' and missing > 0:
            print("Warning: Missing BERT embeddings will be replaced with zeros")
        elif col in ['Sentiment_Polarity', 'Sentiment_Subjectivity'] and missing > 0:
            print(f"Filling missing {col} values with median")
            df[col] = df[col].fillna(df[col].median())
    
    return df

# Process BERT embeddings
def process_embeddings(df):
    """Process BERT embeddings from string format to numpy arrays"""
    # First pass: determine embedding dimension
    emb_dim = None
    for emb_str in df['BERT_Embeddings']:
        try:
            # Remove brackets
            emb_str = emb_str.strip('[]')
            # Try splitting by comma first
            if ',' in emb_str:
                values = [float(x) for x in emb_str.split(',')]
            else:
                # If no commas, split by whitespace
                values = [float(x) for x in emb_str.split()]
            
            if emb_dim is None:
                emb_dim = len(values)
                print(f"Detected embedding dimension: {emb_dim}")
            break
        except:
            continue
    
    # If no valid embedding found, use default
    if emb_dim is None:
        emb_dim = 768  # Default BERT base dimension
        print(f"Using default embedding dimension: {emb_dim}")
    
    # Second pass: parse all embeddings
    embeddings_list = []
    error_count = 0
    for emb_str in df['BERT_Embeddings']:
        try:
            # Remove brackets
            emb_str = emb_str.strip('[]')
            # Try splitting by comma first
            if ',' in emb_str:
                values = [float(x) for x in emb_str.split(',')]
            else:
                # If no commas, split by whitespace
                values = [float(x) for x in emb_str.split()]
            
            # Ensure consistent dimension
            if len(values) == emb_dim:
                embeddings_list.append(np.array(values))
            else:
                print(f"Inconsistent embedding dimension: {len(values)}, expected {emb_dim}. Using zeros.")
                embeddings_list.append(np.zeros(emb_dim))
                error_count += 1
        except:
            print("Error parsing embedding, using zeros")
            embeddings_list.append(np.zeros(emb_dim))
            error_count += 1
    
    print(f"Total embedding parsing errors: {error_count}/{len(df)}")
    
    # Convert list to numpy array
    embeddings = np.array(embeddings_list)
    return embeddings

# Feature preparation
def prepare_features(df, embeddings):
    """Prepare features for the model including embeddings and other relevant features"""
    # Additional features that might be useful (adjust based on your needs)
    features = pd.DataFrame()
    
    # Add sentiment features
    features['Sentiment_Polarity'] = df['Sentiment_Polarity']
    features['Sentiment_Subjectivity'] = df['Sentiment_Subjectivity']
    
    # Optional: Add one-hot encoded sector information
    if 'Sector' in df.columns:
        sector_dummies = pd.get_dummies(df['Sector'], prefix='Sector')
        features = pd.concat([features, sector_dummies], axis=1)
    
    # Convert to numpy and combine with embeddings
    features_array = features.values
    
    # Normalize non-embedding features
    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)
    
    # Combine with embeddings
    # We'll concatenate along axis 1 (columns)
    X = np.hstack((features_array, embeddings))
    
    return X, scaler

# Prepare target variable
def prepare_target(df):
    """Prepare the target variable (% change in stock price)"""
    y = df['Change (%)'].values
    return y

# Split the data
def split_data(X, y, test_size=0.2, val_size=0.1):
    """Split data into training, validation, and test sets"""
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Second split: training vs validation
    # Calculate the proportion of validation data relative to training + validation
    val_proportion = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_proportion, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
#STEP 2------------------------------------------------------------------------------------------------------
# Define the Bayesian Neural Network model
def create_bnn_model(input_shape, prior_stddev=1.0):
    """Create a Bayesian Neural Network model using TensorFlow Probability"""
    
    # Define the prior distribution for the weights
    def prior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        prior_model = tfp.distributions.Normal(
            loc=tf.zeros(n, dtype=dtype),
            scale=tf.ones(n, dtype=dtype) * prior_stddev
        )
        return prior_model
    
    # Define the posterior distribution for the weights
    def posterior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        posterior_model = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(
                loc=t[..., :n//2],
                scale=tf.nn.softplus(t[..., n//2:])
            )
        )
        return posterior_model
    
    # Build the model
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(input_shape,)),
        
        # Hidden layers with Bayesian weights
        tfp.layers.DenseVariational(
            units=128,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1/1000,  # Scale the KL divergence to balance with the data fit term
            activation='relu'
        ),
        
        tfp.layers.DenseVariational(
            units=64,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1/1000,
            activation='relu'
        ),
        
        # Output layer - using a normal distribution to model aleatoric uncertainty
        tfp.layers.DenseVariational(
            units=2,  # Output mean and log_variance
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1/1000,
            activation=None  # No activation for the output
        ),
        
        # Convert the output to a normal distribution
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(
                loc=t[..., :1],  # Mean of the prediction
                scale=tf.nn.softplus(t[..., 1:]) + 1e-6  # Standard deviation must be positive
            )
        )
    ])
    
    return model

# Define a negative log likelihood loss function for the BNN
def negative_log_likelihood(y, model_distribution):
    """Negative log likelihood loss function for probabilistic outputs"""
    return -model_distribution.log_prob(y)

# Function to compile and train the BNN model
def train_bnn_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Compile and train the BNN model with error handling"""
    
    # Print shapes for debugging
    print(f"Training shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Validation shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # Check for NaN values
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        print("Warning: NaN values detected in training data! Fixing...")
        # Replace NaN with zeros in X_train
        X_train = np.nan_to_num(X_train, nan=0.0)
        # Replace NaN with median in y_train
        nan_mask = np.isnan(y_train)
        if nan_mask.any():
            median_y = np.nanmedian(y_train)
            y_train[nan_mask] = median_y
    
    if np.isnan(X_val).any() or np.isnan(y_val).any():
        print("Warning: NaN values detected in validation data! Fixing...")
        X_val = np.nan_to_num(X_val, nan=0.0)
        nan_mask = np.isnan(y_val)
        if nan_mask.any():
            median_y = np.nanmedian(y_val)
            y_val[nan_mask] = median_y
    
    # Compile the model with the negative log likelihood loss
    try:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=negative_log_likelihood
        )
        
        # Define callbacks with error handling
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_bnn_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        # Train the model with try-except
        try:
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            return model, history
        except Exception as e:
            print(f"Error during model training: {e}")
            # Try with a smaller batch size if we get OOM error
            if 'resource exhausted' in str(e).lower():
                print("Attempting to train with smaller batch size...")
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size // 2,  # Reduce batch size
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=1
                )
                return model, history
            else:
                raise e
    except Exception as e:
        print(f"Error compiling model: {e}")
        raise e

# Function to make predictions with uncertainty
def predict_with_uncertainty(model, X, n_samples=100):
    """Generate predictions with uncertainty estimates"""
    
    # Collect samples from the posterior predictive distribution
    predictions = []
    for _ in range(n_samples):
        y_pred = model(X)
        predictions.append(y_pred.mean().numpy())
    
    # Stack the predictions
    predictions = np.stack(predictions, axis=0)
    
    # Calculate the mean prediction and uncertainty
    mean_prediction = np.mean(predictions, axis=0)
    std_prediction = np.std(predictions, axis=0)
    
    return mean_prediction, std_prediction
#step 3------------------------------------------------------------------------------------------------------
# Function to evaluate the model
def evaluate_model(model, X_test, y_test, n_samples=100):
    """Evaluate the BNN model on test data"""
    
    # Get predictions with uncertainty
    mean_pred, std_pred = predict_with_uncertainty(model, X_test, n_samples)
    
    # Calculate metrics
    mse = np.mean((mean_pred.flatten() - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(mean_pred.flatten() - y_test))
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    # Calculate correlation between prediction uncertainty and error
    abs_errors = np.abs(mean_pred.flatten() - y_test)
    correlation = np.corrcoef(std_pred.flatten(), abs_errors)[0, 1]
    print(f"Correlation between uncertainty and error: {correlation:.4f}")
    
    return mse, rmse, mae, correlation

# Function to visualize predictions vs actual values
def visualize_predictions(y_test, mean_pred, std_pred):
    """Visualize the predictions vs actual values with uncertainty"""
    
    plt.figure(figsize=(12, 6))
    
    # Sort by actual values for a cleaner plot
    indices = np.argsort(y_test)
    sorted_y_test = y_test[indices]
    sorted_mean_pred = mean_pred.flatten()[indices]
    sorted_std_pred = std_pred.flatten()[indices]
    
    # Plot actual vs predicted with error bars for uncertainty
    plt.errorbar(range(len(sorted_y_test)), sorted_mean_pred, 
                 yerr=sorted_std_pred, fmt='o', alpha=0.6, 
                 label='Predicted with uncertainty')
    plt.plot(range(len(sorted_y_test)), sorted_y_test, 'ro', label='Actual')
    
    plt.xlabel('Samples (sorted by actual values)')
    plt.ylabel('Stock Price Change (%)')
    plt.title('BNN Predictions vs Actual Values with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bnn_predictions.png')
    plt.show()
    
    # Plot uncertainty vs absolute error
    plt.figure(figsize=(10, 6))
    abs_errors = np.abs(mean_pred.flatten() - y_test)
    
    plt.scatter(sorted_std_pred, abs_errors[indices], alpha=0.6)
    plt.xlabel('Prediction Uncertainty (std)')
    plt.ylabel('Absolute Error')
    plt.title('Uncertainty vs Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('uncertainty_vs_error.png')
    plt.show()

# Function to analyze stocks with high uncertainty
def analyze_high_uncertainty(df, mean_pred, std_pred, threshold=0.9):
    """Analyze stocks with high prediction uncertainty"""
    
    # Calculate the percentile for uncertainty threshold
    uncertainty_threshold = np.percentile(std_pred, threshold * 100)
    
    # Find indices where uncertainty is high
    high_uncertainty_indices = np.where(std_pred.flatten() > uncertainty_threshold)[0]
    
    # Extract information for these high uncertainty predictions
    high_uncertainty_info = []
    for idx in high_uncertainty_indices:
        info = {
            'Symbol': df.iloc[idx]['Symbol'],
            'Company': df.iloc[idx]['Company Name'],
            'Sector': df.iloc[idx]['Sector'],
            'Article_Title': df.iloc[idx]['Article_Title'],
            'Sentiment_Polarity': df.iloc[idx]['Sentiment_Polarity'],
            'Actual_Change': df.iloc[idx]['Change (%)'],
            'Predicted_Change': mean_pred.flatten()[idx],
            'Uncertainty': std_pred.flatten()[idx]
        }
        high_uncertainty_info.append(info)
    
    # Convert to DataFrame for better visualization
    high_uncertainty_df = pd.DataFrame(high_uncertainty_info)
    
    # Sort by uncertainty (highest first)
    high_uncertainty_df = high_uncertainty_df.sort_values('Uncertainty', ascending=False)
    
    return high_uncertainty_df

# Main execution function
def main(filepath):
    """Main function to run the BNN implementation"""
    
    print("Loading and preparing data...")
    # Load the data
    df = load_data(filepath)
    
    # Process BERT embeddings
    embeddings = process_embeddings(df)
    
    # Prepare features and target
    X, scaler = prepare_features(df, embeddings)
    y = prepare_target(df)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Create the BNN model
    input_shape = X_train.shape[1]
    print(f"Creating BNN model with input shape: {input_shape}")
    model = create_bnn_model(input_shape)
    
    # Print model summary
    model.summary()
    
    # Train the model
    print("Training the BNN model...")
    model, history = train_bnn_model(model, X_train, y_train, X_val, y_val, epochs=100)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_history.png')
    plt.show()
    
    # Evaluate the model
    print("Evaluating the model...")
    mse, rmse, mae, correlation = evaluate_model(model, X_test, y_test)
    
    # Make predictions with uncertainty
    print("Generating predictions with uncertainty...")
    mean_pred, std_pred = predict_with_uncertainty(model, X_test)
    
    # Visualize predictions
    visualize_predictions(y_test, mean_pred, std_pred)
    
    # Analyze high uncertainty predictions
    print("Analyzing high uncertainty predictions...")
    high_uncertainty_df = analyze_high_uncertainty(df.iloc[-len(X_test):].reset_index(), mean_pred, std_pred)
    print(high_uncertainty_df.head(10))
    
    # Save the high uncertainty analysis
    high_uncertainty_df.to_csv('high_uncertainty_stocks.csv', index=False)
    
    return model, mean_pred, std_pred

# If running as a script
if __name__ == "__main__":
    # Specify the path to your CSV file
    filepath = "stock_news_data.csv"  # Replace with your actual file path
    
    # Run the BNN implementation
    model, mean_pred, std_pred = main(filepath)
#STEP 4------------------------------------------------------------------------------------------------------
# Function to analyze confidence calibration
def analyze_calibration(y_test, mean_pred, std_pred, n_bins=10):
    """Analyze the calibration of uncertainty estimates"""
    
    # Calculate absolute errors
    abs_errors = np.abs(mean_pred.flatten() - y_test)
    
    # Create bins based on predicted uncertainty (std)
    bins = np.linspace(np.min(std_pred), np.max(std_pred), n_bins + 1)
    indices = np.digitize(std_pred.flatten(), bins) - 1
    
    # Calculate mean error and mean uncertainty for each bin
    mean_errors = []
    mean_uncertainties = []
    counts = []
    
    for i in range(n_bins):
        bin_indices = (indices == i)
        if np.sum(bin_indices) > 0:
            mean_errors.append(np.mean(abs_errors[bin_indices]))
            mean_uncertainties.append(np.mean(std_pred.flatten()[bin_indices]))
            counts.append(np.sum(bin_indices))
        else:
            mean_errors.append(0)
            mean_uncertainties.append(0)
            counts.append(0)
    
    # Plot calibration
    plt.figure(figsize=(10, 6))
    
    # Size points by the number of samples in each bin
    sizes = 100 * np.array(counts) / np.max(counts)
    
    plt.scatter(mean_uncertainties, mean_errors, s=sizes, alpha=0.7)
    
    # Add diagonal line (perfect calibration)
    max_val = max(np.max(mean_uncertainties), np.max(mean_errors))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect calibration')
    
    plt.xlabel('Mean Predicted Uncertainty (std)')
    plt.ylabel('Mean Absolute Error')
    plt.title('Calibration Plot: Uncertainty vs Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('calibration_plot.png')
    plt.show()
    
    # Calculate calibration score (how close to diagonal)
    # Lower is better (0 = perfect calibration)
    calibration_score = np.mean(np.abs(np.array(mean_uncertainties) - np.array(mean_errors)))
    print(f"Calibration Score: {calibration_score:.4f} (lower is better)")
    
    return calibration_score

# Function to analyze sentiment vs prediction accuracy
def analyze_sentiment_vs_accuracy(df, mean_pred, y_test):
    """Analyze how sentiment relates to prediction accuracy"""
    
    # Get sentiment polarity for test samples
    test_indices = df.index[-len(y_test):]
    sentiment_polarities = df.loc[test_indices, 'Sentiment_Polarity'].values
    
    # Calculate absolute errors
    abs_errors = np.abs(mean_pred.flatten() - y_test)
    
    # Prepare data for visualization
    data = pd.DataFrame({
        'Sentiment_Polarity': sentiment_polarities,
        'Absolute_Error': abs_errors,
        'Actual_Change': y_test,
        'Predicted_Change': mean_pred.flatten()
    })
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Sentiment_Polarity'], data['Absolute_Error'], 
                c=data['Actual_Change'], cmap='coolwarm', alpha=0.7)
    
    plt.colorbar(label='Actual Stock Price Change (%)')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Prediction Error')
    plt.title('Sentiment Polarity vs Prediction Error')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sentiment_vs_error.png')
    plt.show()
    
    # Calculate correlation
    correlation = np.corrcoef(sentiment_polarities, abs_errors)[0, 1]
    print(f"Correlation between sentiment polarity and prediction error: {correlation:.4f}")
    
    return correlation

# Function to analyze sector-wise performance
def analyze_sector_performance(df, mean_pred, y_test):
    """Analyze how the model performs across different sectors"""
    
    # Get sectors for test samples
    test_indices = df.index[-len(y_test):]
    sectors = df.loc[test_indices, 'Sector'].values
    
    # Calculate absolute errors
    abs_errors = np.abs(mean_pred.flatten() - y_test)
    
    # Create DataFrame for analysis
    sector_data = pd.DataFrame({
        'Sector': sectors,
        'Absolute_Error': abs_errors,
        'Actual_Change': y_test,
        'Predicted_Change': mean_pred.flatten()
    })
    
    # Group by sector and calculate metrics
    sector_performance = sector_data.groupby('Sector').agg({
        'Absolute_Error': ['mean', 'std', 'count'],
        'Actual_Change': ['mean', 'std'],
        'Predicted_Change': ['mean', 'std']
    })
    
    # Flatten column names
    sector_performance.columns = ['_'.join(col).strip() for col in sector_performance.columns.values]
    
    # Sort by mean absolute error (ascending)
    sector_performance = sector_performance.sort_values('Absolute_Error_mean')
    
    # Visualize sector performance
    plt.figure(figsize=(12, 8))
    
    # Plot mean absolute error by sector
    plt.barh(sector_performance.index, sector_performance['Absolute_Error_mean'], 
             xerr=sector_performance['Absolute_Error_std'], alpha=0.7)
    
    plt.xlabel('Mean Absolute Error')
    plt.ylabel('Sector')
    plt.title('Prediction Error by Sector')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sector_performance.png')
    plt.show()
    
    return sector_performance

# Function to detect potential anomalies
def detect_anomalies(df, mean_pred, std_pred, y_test, threshold=2.0):
    """Detect potential anomalies where actual change deviates significantly from prediction"""
    
    # Calculate z-scores (how many std devs actual is from predicted)
    z_scores = np.abs((y_test - mean_pred.flatten()) / (std_pred.flatten() + 1e-10))
    
    # Find anomalies (high z-score)
    anomaly_indices = np.where(z_scores > threshold)[0]
    
    # Extract information for these anomalies
    anomaly_info = []
    test_indices = df.index[-len(y_test):]
    
    for idx in anomaly_indices:
        actual_idx = test_indices[idx]
        info = {
            'Symbol': df.loc[actual_idx, 'Symbol'],
            'Company': df.loc[actual_idx, 'Company Name'],
            'Sector': df.loc[actual_idx, 'Sector'],
            'Article_Title': df.loc[actual_idx, 'Article_Title'],
            'Sentiment_Polarity': df.loc[actual_idx, 'Sentiment_Polarity'],
            'Actual_Change': y_test[idx],
            'Predicted_Change': mean_pred.flatten()[idx],
            'Uncertainty': std_pred.flatten()[idx],
            'Z_Score': z_scores[idx]
        }
        anomaly_info.append(info)
    
    # Convert to DataFrame for better visualization
    anomaly_df = pd.DataFrame(anomaly_info)
    
    # Sort by z-score (highest first)
    if not anomaly_df.empty:
        anomaly_df = anomaly_df.sort_values('Z_Score', ascending=False)
        
        # Save anomalies to CSV
        anomaly_df.to_csv('anomalies.csv', index=False)
    
    print(f"Detected {len(anomaly_indices)} anomalies with z-score > {threshold}")
    
    return anomaly_df

# Function to save the trained model and necessary artifacts
def save_model_artifacts(model, scaler, input_shape, output_dir='./model_artifacts'):
    """Save the trained model and necessary artifacts for later use"""
    import os
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the model weights
    model.save_weights(os.path.join(output_dir, 'bnn_model_weights.h5'))
    
    # Save the scaler
    import joblib
    joblib.dump(scaler, os.path.join(output_dir, 'feature_scaler.joblib'))
    
    # Save the input shape
    np.save(os.path.join(output_dir, 'input_shape.npy'), np.array([input_shape]))
    
    print(f"Model artifacts saved to {output_dir}")

# Function to load the saved model and artifacts
def load_model_artifacts(output_dir='./model_artifacts'):
    """Load the saved model and artifacts"""
    import os
    import joblib
    
    # Load the input shape
    input_shape = np.load(os.path.join(output_dir, 'input_shape.npy'))[0]
    
    # Create a new model with the same architecture
    model = create_bnn_model(input_shape)
    
    # Load the weights
    model.load_weights(os.path.join(output_dir, 'bnn_model_weights.h5'))
    
    # Load the scaler
    scaler = joblib.load(os.path.join(output_dir, 'feature_scaler.joblib'))
    
    print(f"Model artifacts loaded from {output_dir}")
    
    return model, scaler
#step5------------------------------------------------------------------------------------------------------
# Full implementation of the Bayesian Neural Network for stock price prediction
def run_full_bnn_analysis(filepath):
    """Run a complete BNN analysis pipeline on the stock data"""
    
    print("="*80)
    print("BAYESIAN NEURAL NETWORK FOR STOCK PRICE PREDICTION")
    print("="*80)
    
    # Step 1: Load and prepare data
    print("\n1. Loading and preparing data...")
    df = load_data(filepath)
    
    # Basic statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of companies: {df['Symbol'].nunique()}")
    print(f"Number of sectors: {df['Sector'].nunique()}")
    print("\nSector distribution:")
    sector_counts = df['Sector'].value_counts()
    for sector, count in sector_counts.items():
        print(f"  - {sector}: {count} ({count/len(df)*100:.1f}%)")
    
    # Process BERT embeddings
    print("\nProcessing BERT embeddings...")
    embeddings = process_embeddings(df)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Prepare features and target
    X, scaler = prepare_features(df, embeddings)
    y = prepare_target(df)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Basic EDA on the target variable
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(y, bins=30, alpha=0.7, color='blue')
    plt.xlabel('Stock Price Change (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Stock Price Changes')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=y)
    plt.ylabel('Stock Price Change (%)')
    plt.title('Box Plot of Stock Price Changes')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('target_distribution.png')
    plt.show()
    
    # Step 2: Create the BNN model
    print("\n2. Creating the BNN model...")
    input_shape = X_train.shape[1]
    model = create_bnn_model(input_shape)
    model.summary()
    
    # Step 3: Train the model
    print("\n3. Training the BNN model...")
    model, history = train_bnn_model(model, X_train, y_train, X_val, y_val, epochs=100)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_history.png')
    plt.show()
    
    # Save model artifacts
    save_model_artifacts(model, scaler, input_shape)
    
    # Step 4: Evaluate the model
    print("\n4. Evaluating the model...")
    mse, rmse, mae, correlation = evaluate_model(model, X_test, y_test)
    
    # Make predictions with uncertainty
    print("\n5. Generating predictions with uncertainty...")
    mean_pred, std_pred = predict_with_uncertainty(model, X_test)
    
    # Visualize predictions
    visualize_predictions(y_test, mean_pred, std_pred)
    
    # Step 5: Analyze calibration
    print("\n6. Analyzing uncertainty calibration...")
    calibration_score = analyze_calibration(y_test, mean_pred, std_pred)
    
    # Step 6: Analyze sentiment vs accuracy
    print("\n7. Analyzing sentiment vs prediction accuracy...")
    sentiment_correlation = analyze_sentiment_vs_accuracy(df, mean_pred, y_test)
    
    # Step 7: Analyze sector performance
    print("\n8. Analyzing sector-wise performance...")
    sector_performance = analyze_sector_performance(df, mean_pred, y_test)
    print("\nSector Performance:")
    print(sector_performance)
    
    # Save sector performance to CSV
    sector_performance.to_csv('sector_performance.csv')
    
    # Step 8: Detect anomalies
    print("\n9. Detecting potential anomalies...")
    anomaly_df = detect_anomalies(df, mean_pred, std_pred, y_test)
    if not anomaly_df.empty:
        print("\nTop 5 Anomalies:")
        print(anomaly_df.head(5))
    
    # Step 9: Analyze high uncertainty predictions
    print("\n10. Analyzing high uncertainty predictions...")
    # Get the corresponding test indices
    test_indices = df.index[-len(X_test):].tolist()
    test_df = df.loc[test_indices].reset_index(drop=True)
    
    high_uncertainty_df = analyze_high_uncertainty(test_df, mean_pred, std_pred)
    print("\nTop 5 High Uncertainty Predictions:")
    print(high_uncertainty_df.head(5))
    
    # Step 10: Generate comprehensive report
    print("\n11. Generating comprehensive analysis report...")
    
    # Create a summary dataframe with key metrics
    summary = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'Error-Uncertainty Correlation', 
                   'Calibration Score', 'Sentiment-Error Correlation'],
        'Value': [mse, rmse, mae, correlation, calibration_score, sentiment_correlation]
    })
    
    # Save summary to CSV
    summary.to_csv('model_performance_summary.csv', index=False)
    
    print("\nSummary of Model Performance:")
    print(summary)
    
    print("\nAnalysis complete! All results have been saved to CSV files and visualizations.")
    
    return model, mean_pred, std_pred, summary

# Function to make new predictions on unseen data
def predict_on_new_data(model, new_data_filepath, model_artifacts_dir='./model_artifacts'):
    """Make predictions on new data using the trained BNN model"""
    
    print("Loading new data...")
    df = load_data(new_data_filepath)
    
    # Process BERT embeddings
    embeddings = process_embeddings(df)
    
    # Load model artifacts
    _, scaler = load_model_artifacts(model_artifacts_dir)
    
    # Prepare features
    # Note: We're only creating features, not target since we're predicting
    features = pd.DataFrame()
    
    # Add sentiment features
    features['Sentiment_Polarity'] = df['Sentiment_Polarity']
    features['Sentiment_Subjectivity'] = df['Sentiment_Subjectivity']
    
    # Optional: Add one-hot encoded sector information
    if 'Sector' in df.columns:
        sector_dummies = pd.get_dummies(df['Sector'], prefix='Sector')
        features = pd.concat([features, sector_dummies], axis=1)
    
    # Convert to numpy and scale
    features_array = features.values
    features_array = scaler.transform(features_array)
    
    # Combine with embeddings
    X = np.hstack((features_array, embeddings))
    
    # Make predictions with uncertainty
    print("Generating predictions with uncertainty...")
    mean_pred, std_pred = predict_with_uncertainty(model, X)
    
    # Add predictions to the dataframe
    df['Predicted_Change'] = mean_pred.flatten()
    df['Prediction_Uncertainty'] = std_pred.flatten()
    
    # Calculate predicted new close price
    df['Predicted_New_Close'] = df['Previous Close'] * (1 + df['Predicted_Change']/100)
    
    # Save predictions to CSV
    df.to_csv('new_predictions.csv', index=False)
    
    print(f"Predictions made on {len(df)} samples and saved to 'new_predictions.csv'")
    
    # Sort by uncertainty and show the most uncertain predictions
    high_uncertainty = df.sort_values('Prediction_Uncertainty', ascending=False)[
        ['Symbol', 'Company Name', 'Previous Close', 'Predicted_Change', 
         'Prediction_Uncertainty', 'Article_Title', 'Sentiment_Polarity']
    ].head(10)
    
    print("\nTop 10 Most Uncertain Predictions:")
    print(high_uncertainty)
    
    return df

# Function to create a trading strategy based on BNN predictions
def create_trading_strategy(predictions_df, confidence_threshold=0.8, change_threshold=2.0):
    """Create a simple trading strategy based on BNN predictions and confidence"""
    
    # Calculate the threshold for low uncertainty (high confidence)
    uncertainty_threshold = np.percentile(predictions_df['Prediction_Uncertainty'], 
                                         (1-confidence_threshold)*100)
    
    # Initialize strategy recommendations
    strategy = pd.DataFrame()
    
    # Find strong sell signals (high confidence in significant negative change)
    strong_sell = predictions_df[
        (predictions_df['Prediction_Uncertainty'] < uncertainty_threshold) & 
        (predictions_df['Predicted_Change'] < -change_threshold)
    ][['Symbol', 'Company Name', 'Previous Close', 'Predicted_Change', 
       'Prediction_Uncertainty', 'Article_Title', 'Sentiment_Polarity']]
    
    # Find strong buy signals (high confidence in significant positive change)
    strong_buy = predictions_df[
        (predictions_df['Prediction_Uncertainty'] < uncertainty_threshold) & 
        (predictions_df['Predicted_Change'] > change_threshold)
    ][['Symbol', 'Company Name', 'Previous Close', 'Predicted_Change', 
       'Prediction_Uncertainty', 'Article_Title', 'Sentiment_Polarity']]
    
    print(f"\nTrading Strategy (Confidence: {confidence_threshold*100}%, Change Threshold: {change_threshold}%)")
    print(f"Found {len(strong_buy)} strong buy signals and {len(strong_sell)} strong sell signals")
    
    if len(strong_buy) > 0:
        print("\nTop 5 Buy Recommendations:")
        print(strong_buy.sort_values('Predicted_Change', ascending=False).head(5))
        
    if len(strong_sell) > 0:
        print("\nTop 5 Sell Recommendations:")
        print(strong_sell.sort_values('Predicted_Change', ascending=True).head(5))
    
    # Save strategy to CSV
    strategy = pd.concat([
        strong_buy.assign(Recommendation='Strong Buy'),
        strong_sell.assign(Recommendation='Strong Sell')
    ])
    
    strategy.to_csv('trading_strategy.csv', index=False)
    
    return strategy

# Helper function for interactive analysis
def interactive_analysis(model, df):
    """Perform an interactive analysis of a specific stock or news article"""
    
    import time
    
    print("\nInteractive Stock Analysis Tool")
    print("===============================")
    
    while True:
        print("\nOptions:")
        print("1. Analyze a specific stock symbol")
        print("2. Analyze a sector")
        print("3. Analyze a specific news article")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            symbol = input("Enter stock symbol to analyze: ").upper()
            
            # Filter data for this symbol
            stock_data = df[df['Symbol'] == symbol]
            
            if len(stock_data) == 0:
                print(f"No data found for symbol {symbol}")
                continue
            
            print(f"\nFound {len(stock_data)} records for {symbol}")
            print(f"Company Name: {stock_data['Company Name'].iloc[0]}")
            print(f"Sector: {stock_data['Sector'].iloc[0]}")
            
            # Process and predict for this stock
            # (This is simplified - in a real app, you'd need to properly process the data)
            embeddings = process_embeddings(stock_data)
            X, _ = prepare_features(stock_data, embeddings)
            
            # Make prediction
            mean_pred, std_pred = predict_with_uncertainty(model, X)
            
            # Display results
            print("\nPrediction Results:")
            for i in range(len(stock_data)):
                print(f"\nArticle: {stock_data['Article_Title'].iloc[i]}")
                print(f"Date: {stock_data['Article_Date'].iloc[i]}")
                print(f"Sentiment Polarity: {stock_data['Sentiment_Polarity'].iloc[i]:.2f}")
                print(f"Previous Close: ${stock_data['Previous Close'].iloc[i]:.2f}")
                print(f"Predicted Change: {mean_pred[i][0]:.2f}% ± {std_pred[i][0]:.2f}%")
                predicted_price = stock_data['Previous Close'].iloc[i] * (1 + mean_pred[i][0]/100)
                print(f"Predicted New Close: ${predicted_price:.2f}")
                
                # If we have actual data
                if 'New Close' in stock_data.columns:
                    actual_change = ((stock_data['New Close'].iloc[i] - stock_data['Previous Close'].iloc[i]) / 
                                     stock_data['Previous Close'].iloc[i] * 100)
                    print(f"Actual Change: {actual_change:.2f}%")
                    print(f"Actual New Close: ${stock_data['New Close'].iloc[i]:.2f}")
            
        elif choice == '2':
            print("\nAvailable sectors:")
            sectors = df['Sector'].unique()
            for i, sector in enumerate(sectors):
                print(f"{i+1}. {sector}")
            
            sector_idx = int(input("\nSelect sector number: ")) - 1
            if sector_idx < 0 or sector_idx >= len(sectors):
                print("Invalid sector number")
                continue
            
            selected_sector = sectors[sector_idx]
            print(f"\nAnalyzing sector: {selected_sector}")
            
            # Filter data for this sector
            sector_data = df[df['Sector'] == selected_sector]
            print(f"Found {len(sector_data)} records in {selected_sector} sector")
            
            # Calculate mean performance if we have actual data
            if 'Change (%)' in sector_data.columns:
                mean_change = sector_data['Change (%)'].mean()
                print(f"Mean actual stock price change: {mean_change:.2f}%")
            
            # Get predictions for this sector
            embeddings = process_embeddings(sector_data)
            X, _ = prepare_features(sector_data, embeddings)
            
            # Make prediction
            mean_pred, std_pred = predict_with_uncertainty(model, X)
            
            # Calculate average predicted change
            avg_pred_change = np.mean(mean_pred)
            avg_uncertainty = np.mean(std_pred)
            
            print(f"Average predicted change: {avg_pred_change:.2f}% ± {avg_uncertainty:.2f}%")
            
            # Show top 5 stocks with highest predicted growth
            sector_data = sector_data.copy()
            sector_data['Predicted_Change'] = mean_pred.flatten()
            sector_data['Prediction_Uncertainty'] = std_pred.flatten()
            
            top_growth = sector_data.sort_values('Predicted_Change', ascending=False)[
                ['Symbol', 'Company Name', 'Previous Close', 'Predicted_Change', 
                 'Prediction_Uncertainty', 'Article_Title']
            ].head(5)
            
            print("\nTop 5 Stocks with Highest Predicted Growth:")
            print(top_growth)
            
        elif choice == '3':
            # Simple keyword search in article titles
            keyword = input("Enter keyword to search in article titles: ")
            
            # Filter articles containing keyword
            articles = df[df['Article_Title'].str.contains(keyword, case=False, na=False)]
            
            if len(articles) == 0:
                print(f"No articles found containing '{keyword}'")
                continue
            
            print(f"\nFound {len(articles)} articles containing '{keyword}'")
            
            # List articles for selection
            for i, (_, row) in enumerate(articles.iterrows()):
                print(f"{i+1}. {row['Symbol']} - {row['Article_Title']}")
            
            article_idx = int(input("\nSelect article number to analyze: ")) - 1
            if article_idx < 0 or article_idx >= len(articles):
                print("Invalid article number")
                continue
            
            selected_article = articles.iloc[article_idx]
            
            print(f"\nAnalyzing article: {selected_article['Article_Title']}")
            print(f"Company: {selected_article['Company Name']} ({selected_article['Symbol']})")
            print(f"Date: {selected_article['Article_Date']}")
            print(f"Sentiment: {selected_article['Sentiment_Polarity']:.2f}")
            
            # Process and predict for this article
            article_df = pd.DataFrame([selected_article])
            embeddings = process_embeddings(article_df)
            X, _ = prepare_features(article_df, embeddings)
            
            # Make prediction
            mean_pred, std_pred = predict_with_uncertainty(model, X)
            
            print(f"\nPredicted Change: {mean_pred[0][0]:.2f}% ± {std_pred[0][0]:.2f}%")
            predicted_price = selected_article['Previous Close'] * (1 + mean_pred[0][0]/100)
            print(f"Previous Close: ${selected_article['Previous Close']:.2f}")
            print(f"Predicted New Close: ${predicted_price:.2f}")
            
            # If we have actual data
            if 'New Close' in selected_article:
                actual_change = ((selected_article['New Close'] - selected_article['Previous Close']) / 
                                 selected_article['Previous Close'] * 100)
                print(f"Actual Change: {actual_change:.2f}%")
                print(f"Actual New Close: ${selected_article['New Close']:.2f}")
            
            # Show article snippet
            print("\nArticle Snippet:")
            print(selected_article['Article_Snippet'])
            
        elif choice == '4':
            print("Exiting interactive analysis...")
            break
        
        else:
            print("Invalid choice, please try again")
        
        time.sleep(1)

# Main entry point
if __name__ == "__main__":
    # Specify the path to your CSV file
    filepath = "stock_news_data.csv"  # Replace with your actual file path
    
    # Run the complete BNN analysis
    model, mean_pred, std_pred, summary = run_full_bnn_analysis(filepath)
    
    # Optional: Make predictions on new data
    # new_predictions = predict_on_new_data(model, "new_stock_data.csv")
    
    # Optional: Create trading strategy based on predictions
    # strategy = create_trading_strategy(new_predictions)
    
    # Optional: Run interactive analysis
    # interactive_analysis(model, pd.read_csv(filepath))
    
    print("\nAnalysis complete!")
# Import all previously defined functions
# (In a real implementation, you would import them from your modules)

def parse_arguments():
    """Parse command-line arguments for the BNN tool"""
    parser = argparse.ArgumentParser(
        description='Bayesian Neural Network for Stock Price Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new BNN model')
    train_parser.add_argument('--data', type=str, required=True, 
                             help='Path to the CSV data file')
    train_parser.add_argument('--output', type=str, default='./model_artifacts',
                             help='Directory to save model artifacts')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size for training')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                             help='Proportion of data to use for testing')
    train_parser.add_argument('--val-size', type=float, default=0.1,
                             help='Proportion of training data to use for validation')
    train_parser.add_argument('--no-plots', action='store_true',
                             help='Disable plotting visualizations')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions using a trained model')
    predict_parser.add_argument('--model', type=str, required=True,
                               help='Directory containing model artifacts')
    predict_parser.add_argument('--data', type=str, required=True,
                               help='Path to the CSV data file to make predictions on')
    predict_parser.add_argument('--output', type=str, default='predictions.csv',
                               help='Path to save predictions')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run interactive analysis')
    analyze_parser.add_argument('--model', type=str, required=True,
                               help='Directory containing model artifacts')
    analyze_parser.add_argument('--data', type=str, required=True,
                               help='Path to the CSV data file to analyze')
    
    # Strategy command
    strategy_parser = subparsers.add_parser('strategy', help='Generate trading strategy')
    strategy_parser.add_argument('--predictions', type=str, required=True,
                                help='Path to predictions CSV file')
    strategy_parser.add_argument('--confidence', type=float, default=0.8,
                                help='Confidence threshold (0-1)')
    strategy_parser.add_argument('--change', type=float, default=2.0,
                                help='Minimum percentage change threshold')
    strategy_parser.add_argument('--output', type=str, default='strategy.csv',
                                help='Path to save strategy recommendations')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    evaluate_parser.add_argument('--model', type=str, required=True,
                                help='Directory containing model artifacts')
    evaluate_parser.add_argument('--data', type=str, required=True,
                                help='Path to the CSV data file to evaluate on')
    evaluate_parser.add_argument('--output', type=str, default='evaluation.csv',
                                help='Path to save evaluation results')
    
    return parser.parse_args()

def main():
    """Main entry point for the BNN command-line tool"""
    args = parse_arguments()
    
    if args.command == 'train':
        print(f"Training BNN model using data from {args.data}")
        print(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}")
        print(f"Train/val/test split: {1-args.test_size-args.val_size}/{args.val_size}/{args.test_size}")
        
        # Load and prepare data
        df = load_data(args.data)
        embeddings = process_embeddings(df)
        X, scaler = prepare_features(df, embeddings)
        y = prepare_target(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, test_size=args.test_size, val_size=args.val_size
        )
        
        # Create and train model
        input_shape = X_train.shape[1]
        model = create_bnn_model(input_shape)
        model, history = train_bnn_model(
            model, X_train, y_train, X_val, y_val, 
            epochs=args.epochs, batch_size=args.batch_size
        )
        
        # Evaluate model
        mse, rmse, mae, correlation = evaluate_model(model, X_test, y_test)
        
        # Save model artifacts
        save_model_artifacts(model, scaler, input_shape, args.output)
        
        # Generate visualizations if not disabled
        if not args.no_plots:
            mean_pred, std_pred = predict_with_uncertainty(model, X_test)
            visualize_predictions(y_test, mean_pred, std_pred)
            analyze_calibration(y_test, mean_pred, std_pred)
        
        print(f"\nTraining complete! Model artifacts saved to {args.output}")
        print(f"Model performance: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
    elif args.command == 'predict':
        print(f"Making predictions using model from {args.model} on data from {args.data}")
        
        # Load model
        model, scaler = load_model_artifacts(args.model)
        
        # Make predictions
        df_with_predictions = predict_on_new_data(model, args.data, args.model)
        
        # Save predictions
        df_with_predictions.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
        
    elif args.command == 'analyze':
        print(f"Running interactive analysis using model from {args.model} on data from {args.data}")
        
        # Load model
        model, _ = load_model_artifacts(args.model)
        
        # Load data
        df = pd.read_csv(args.data)
        
        # Run interactive analysis
        interactive_analysis(model, df)
        
    elif args.command == 'strategy':
        print(f"Generating trading strategy from predictions in {args.predictions}")
        print(f"Using confidence threshold: {args.confidence}, change threshold: {args.change}%")
        
        # Load predictions
        predictions_df = pd.read_csv(args.predictions)
        
        # Create strategy
        strategy = create_trading_strategy(
            predictions_df, 
            confidence_threshold=args.confidence,
            change_threshold=args.change
        )
        
        # Save strategy
        strategy.to_csv(args.output, index=False)
        print(f"Strategy recommendations saved to {args.output}")
        
    elif args.command == 'evaluate':
        print(f"Evaluating model from {args.model} on data from {args.data}")
        
        # Load model
        model, scaler = load_model_artifacts(args.model)
        
        # Load and prepare data
        df = load_data(args.data)
        embeddings = process_embeddings(df)
        X, _ = prepare_features(df, embeddings)
        y = prepare_target(df)
        
        # Evaluate model
        mse, rmse, mae, correlation = evaluate_model(model, X, y)
        
        # Generate predictions with uncertainty
        mean_pred, std_pred = predict_with_uncertainty(model, X)
        
        # Analyze calibration
        calibration_score = analyze_calibration(y, mean_pred, std_pred)
        
        # Create summary dataframe
        summary = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'Error-Uncertainty Correlation', 'Calibration Score'],
            'Value': [mse, rmse, mae, correlation, calibration_score]
        })
        
        # Save summary
        summary.to_csv(args.output, index=False)
        print(f"Evaluation results saved to {args.output}")
        print(f"Model performance: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
    else:
        print("Please specify a command. Use --help for options.")



# temp main function ------------------------------------------------------------------------

if __name__ == "__main__":
    main()
