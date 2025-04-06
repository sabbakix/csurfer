import pandas as pd
import mplfinance as mpf

def plot_candlestick_chart(csv_file):
    """
    Reads price data from a CSV file and plots a candlestick chart.
    The CSV file should have columns: 'timestamp', 'open', 'high', 'low', 'close'.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Ensure the DataFrame has the required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

    # Convert 'timestamp' to a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Plot the candlestick chart
    mpf.plot(df, type='candle', style='charles', title='Candlestick Chart', ylabel='Price')

# Example usage
csv_file = 'data/btcusd_1-min_data.csv'  # Replace with the path to your CSV file
plot_candlestick_chart(csv_file)