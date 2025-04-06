import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def prepare_data(csv_file, lookahead_minutes=30, window_size=300000):
    """
    Prepares the data for training the AI model, keeping a sliding window of past rows.
    Each row corresponds to one minute, and the model predicts `lookahead_minutes` into the future.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Ensure the DataFrame has the required columns
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

    # Create a target column: 1 if price goes up, 0 if it goes down
    df['target'] = (df['close'].shift(-lookahead_minutes) > df['close']).astype(int)

    # Drop rows with NaN values (caused by the shift)
    df.dropna(inplace=True)

    # Use only the last `window_size` rows
    if len(df) > window_size:
        df = df.iloc[-window_size:]

    # Features and target
    X = df[['open', 'high', 'low', 'close']].values
    y = df['target'].values

    # Normalize the features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Reshape X for LSTM input (samples, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X, y

def build_model(input_shape):
    """
    Builds and compiles an LSTM model.
    """
    model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=False),
        Dense(1, activation='sigmoid')  # Output probability
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_predict(csv_file, lookahead_minutes=30, window_size=300000):
    """
    Trains the model and predicts the probability of price going up or down 30 minutes in advance.
    """
    # Prepare the data
    X, y = prepare_data(csv_file, lookahead_minutes=lookahead_minutes, window_size=window_size)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = build_model(X_train.shape[1:])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Predict probabilities on the test set
    predictions = model.predict(X_test)
    return predictions

# Example usage
csv_file = 'data/btcusd_1-min_data.csv'  # Replace with the path to your CSV file
csv_file = 'data/btcusd_1-min_data_1650470340.0_data_after.csv'

# Predict for 30 minutes in advance
predictions = train_and_predict(csv_file, lookahead_minutes=30)

# Print the first 30 predictions
print("Predictions (probability of price going up):", predictions[:30])