import websocket
import json
import os
import csv

# Define the CSV file path
csv_file = "data/binance_data_stream.csv"
# Ensure the CSV file exists and has a header
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["symbol", "price", "time"])  # Add header row

# Define the CSV file path
csv_file2 = "data/binance_data.csv"
# Ensure the CSV file exists and has a header
if not os.path.exists(csv_file2):
    with open(csv_file2, mode='w', newline='', encoding='utf-8') as file:
        writer2 = csv.writer(file, delimiter='§')
        writer2.writerow(["Response"])  # Add header row




def on_message(ws, message):
    data = json.loads(message)
    if 'stream' in data:
        # you can pipeline this data to your function, analysis or backtesting
        print(f"Symbol: {data['data']['s']}, Price: {data['data']['c']}, Time: {data['data']['E']}")
        # Extract relevant data
        symbol = data['data']['s']
        price = data['data']['c']
        timestamp = data['data']['E']

        # Append the data to the CSV file
        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([symbol, price, timestamp])
    else:
        print(f"Received message: {message}")
        # Append the data to the CSV file
        with open(csv_file2, mode='a', newline='', encoding='utf-8') as file:
            writer2 = csv.writer(file)
            writer2.writerow([message])



def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

def on_open(ws):
    print("WebSocket connection opened")
    # Subscribe to the ticker stream for BTCUSDT
    subscribe_message = {
        "method": "SUBSCRIBE",
        #"params": ["btcusdt@ticker"],
        #"params": ["btcusdt@aggTrade"],
        #"params": ["btcusdt@depth"],
        #"params": ["!ticker@arr"],
         
        "params": [
            "btcusdt@ticker",
            "btcusdt@aggTrade",
            "btcusdt@depth",
            "!ticker@arr"
            ],
         
        "id": 1
    }
    ws.send(json.dumps(subscribe_message))

def on_ping(ws, message):
    print(f"Received ping: {message}")
    ws.send(message, websocket.ABNF.OPCODE_PONG)
    print(f"Sent pong: {message}")

if __name__ == "__main__":
    #websocket.enableTrace(True)
    socket = 'wss://stream.binance.com:9443/ws'
    ws = websocket.WebSocketApp(socket,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open,
                                on_ping=on_ping)
    ws.run_forever()