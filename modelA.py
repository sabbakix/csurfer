import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def prepare_data(csv_file, lookahead=30):
    """
    Prepares the data for training the AI model.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Ensure the DataFrame has the required columns
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

    # Create a target column: 1 if price goes up, 0 if it goes down
    df['target'] = (df['close'].shift(-lookahead) > df['close']).astype(int)

    # Drop rows with NaN values (caused by the shift)
    df.dropna(inplace=True)

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

def train_and_predict(csv_file):
    """
    Trains the model and predicts the probability of price going up or down.
    """
    # Prepare the data
    X, y = prepare_data(csv_file)

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
predictions = train_and_predict(csv_file)

# Print the first 10 predictions
print("Predictions (probability of price going up):", predictions[:10])