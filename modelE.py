import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten

def calculate_indicators(df):
    """
    Calculates various indicators and adds them as new columns to the DataFrame.
    """
    # Moving Averages
    df['ma_10'] = df['close'].rolling(window=10).mean()  # 10-period moving average
    df['ma_30'] = df['close'].rolling(window=30).mean()  # 30-period moving average

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Exponential Moving Average (EMA)
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

    # Bollinger Bands
    df['bb_upper'] = df['ma_10'] + 2 * df['close'].rolling(window=10).std()
    df['bb_lower'] = df['ma_10'] - 2 * df['close'].rolling(window=10).std()

    # Drop rows with NaN values caused by rolling calculations
    df.dropna(inplace=True)
    return df

def prepare_data(csv_file, lookahead_minutes=30, window_size=300000):
    """
    Prepares the data for training the AI model, including precalculated indicators.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Ensure the DataFrame has the required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

    # Calculate indicators
    df = calculate_indicators(df)

    # Create a target column: 1 if price goes up, 0 if it goes down
    df['target'] = (df['close'].shift(-lookahead_minutes) > df['close']).astype(int)

    # Drop rows with NaN values (caused by the shift)
    df.dropna(inplace=True)

    # Use only the last `window_size` rows
    if len(df) > window_size:
        df = df.iloc[-window_size:]

    # Features and target
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'ma_10', 'ma_30', 'rsi', 'ema_10', 'bb_upper', 'bb_lower']
    X = df[feature_columns].values
    y = df['target'].values

    # Normalize the features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Reshape X for LSTM input (samples, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X, y

def build_model(input_shape):
    """
    Builds and compiles a hybrid LSTM + DNN model.
    """
    model = Sequential([
        # LSTM layers
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),

        # Dense layers
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Output probability
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_predict(csv_file, lookahead_minutes=30, window_size=300000):
    """
    Trains the model and predicts the probability of price going up or down.
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

    # Return predictions and test data for evaluation
    return predictions, X_test, y_test

# Example usage
csv_file = 'data/btcusd_1-min_data.csv'  # Replace with the path to your CSV file
predictions, X_test, y_test = train_and_predict(csv_file, lookahead_minutes=30)

# Print the first 10 predictions
print("Predictions (probability of price going up):", predictions[:10].flatten())