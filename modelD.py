import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten

def calculate_rsi(data, window=14):
    """
    Calculates the Relative Strength Index (RSI).
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_data(csv_file, lookahead_minutes=30, window_size=300000):
    """
    Prepares the data for training the AI model, keeping a sliding window of past rows.
    Each row corresponds to one minute, and the model predicts `lookahead_minutes` into the future.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Ensure the DataFrame has the required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

    # Calculate Moving Average (MA) and RSI
    df['ma_10'] = df['close'].rolling(window=10).mean()  # 10-period moving average
    df['ma_30'] = df['close'].rolling(window=30).mean()  # 30-period moving average
    df['rsi'] = calculate_rsi(df['close'])

    # Drop rows with NaN values (caused by rolling calculations)
    df.dropna(inplace=True)

    # Create a target column: 1 if price goes up, 0 if it goes down
    df['target'] = (df['close'].shift(-lookahead_minutes) > df['close']).astype(int)

    # Drop rows with NaN values (caused by the shift)
    df.dropna(inplace=True)

    # Use only the last `window_size` rows
    if len(df) > window_size:
        df = df.iloc[-window_size:]

    # Features and target
    X = df[['open', 'high', 'low', 'close', 'volume', 'ma_10', 'ma_30', 'rsi']].values
    y = df['target'].values

    # Normalize the features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Reshape X for LSTM input (samples, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X, y

def build_model(input_shape):
    """
    Builds and compiles a mixed LSTM and Dense Neural Network model.
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Output probability
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_predict(csv_file, lookahead_minutes=30, window_size=300000):
    """
    Trains the model and predicts the probability of price going up or down 30 minutes in advance.
    Includes the price at the prediction time and the future price.
    """
    # Prepare the data
    df = pd.read_csv(csv_file)  # Load the CSV file to access timestamps and prices
    X, y = prepare_data(csv_file, lookahead_minutes=lookahead_minutes, window_size=window_size)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test, timestamps_train, timestamps_test = train_test_split(
        X, y, df['timestamp'].values[-len(X):], test_size=0.2, random_state=42
    )

    # Build the model
    model = build_model(X_train.shape[1:])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Predict probabilities on the test set
    predictions = model.predict(X_test)

    # Articulate results
    results = []
    for i in range(len(predictions)):
        prediction_time = timestamps_test[i]
        target_time = prediction_time + (lookahead_minutes * 60)  # Add 30 minutes in seconds

        # Get the price at the prediction time and the target time
        prediction_price = df.loc[df['timestamp'] == prediction_time, 'close'].values[0]
        future_price = df.loc[df['timestamp'] == target_time, 'close'].values[0] if target_time in df['timestamp'].values else None

        results.append({
            "prediction_time": prediction_time,
            "target_time": target_time,
            "prediction_price": prediction_price,
            "future_price": future_price,
            "probability_up": predictions[i][0]
        })

    return results

# Example usage
csv_file = 'data/btcusd_1-min_data.csv'  # Replace with the path to your CSV file
csv_file = 'data/btcusd_1-min_data_1650470340.0_data_after.csv'

# Predict for 30 minutes in advance
results = train_and_predict(csv_file, lookahead_minutes=30)

# Print the first 10 articulated results
print("Articulated Predictions:")
for result in results[:10]:
    print(f"Prediction Time: {pd.to_datetime(result['prediction_time'], unit='s')}, "
          f"Target Time: {pd.to_datetime(result['target_time'], unit='s')}, "
          f"Prediction Price: {result['prediction_price']}, "
          f"Future Price: {result['future_price']}, "
          f"Probability of Price Going Up: {result['probability_up']}")