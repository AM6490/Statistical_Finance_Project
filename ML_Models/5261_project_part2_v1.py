# -*- coding: utf-8 -*-
"""5261_project_part2_v1.ipynb
"""

import os
import pandas as pd

def load_csvs_to_dict(2018_to_2024):
    data_dict = {}

    # Ensure the folder exists
    if not os.path.exists(2018_to_2024):
        print(f"Error: Folder '{2018_to_2024}' does not exist.")
        return data_dict

    # Iterate through the files in the folder
    for file in os.listdir(2018_to_2024):
        if file.endswith(".csv"):  # Ensure only CSV files are processed
            file_path = os.path.join(2018_to_2024, file)
            df_name = os.path.splitext(file)[0]  # Extract name without extension
            data_dict[df_name] = pd.read_csv(file_path, index_col=0)  # Read CSV with index

    return data_dict

# Test Code
folder = "2018_to_2024"
data_frames1 = load_csvs_to_dict(folder)

# Print the names of the DataFrames
print("Loaded DataFrames:", list(data_frames1.keys()))

# Display the first 5 rows and first 10 columns of each DataFrame
for name, df in data_frames1.items():
    print(f"\nDataFrame: {name}")
    display(df.iloc[:5, :5])  # Display first 5 rows and first 5 columns

import pandas as pd

def combine_columns_from_df_dict(df_dict, n=0):
    """
    Combine the ith column from each DataFrame in df_dict into a new DataFrame.

    Parameters:
    - df_dict (dict of pd.DataFrame): Dictionary of dataframes with the same shape.
    - n (int): Number of columns to iterate over. If 0, use all columns.

    Returns:
    - result_dict (dict of pd.DataFrame): Dictionary with column names as keys and
      DataFrames with ith columns from all input DataFrames as values.
    """
    # Assume all dataframes have the same shape and columns
    first_df = next(iter(df_dict.values()))
    total_cols = len(first_df.columns)
    n = total_cols if n == 0 else min(n, total_cols)

    result_dict = {}

    for i in range(n):
        col_name = first_df.columns[i]
        combined_df = pd.DataFrame({
            key: df.iloc[:, i] for key, df in df_dict.items()
        })
        result_dict[col_name] = combined_df

    return result_dict

data_frames2 = combine_columns_from_df_dict(data_frames1, n=10)

for name, df in data_frames2.items():
    print(f"\nDataFrame: {name}")
    display(df.iloc[:3, :5])  # Display first 5 rows and first 5 columns

import re

def save_df_dict_to_csv(df_dict, folder_name):
    """
    Save each DataFrame in df_dict to a CSV file in the specified folder.

    Parameters:
    - df_dict (dict of pd.DataFrame): Dictionary with DataFrames as values.
    - folder_name (str): Folder where CSV files will be saved.
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Save each DataFrame as a CSV file
    for name, df in df_dict.items():
        # Replace illegal characters in filename
        safe_name = re.sub(r'[\\/:"*?<>|]', '-', name)
        file_path = os.path.join(folder_name, f"{safe_name}.csv")
        df.to_csv(file_path, index=True)


save_df_dict_to_csv(data_frames2, "2018_to_2024_by_companies")

# Test Code
folder = "2018_to_2024_by_companies"
data_frames3 = load_csvs_to_dict(folder)

# Print the names of the DataFrames
print("Loaded DataFrames:", list(data_frames3.keys()))

# Display the first 5 rows and first 10 columns of each DataFrame
for name, df in data_frames3.items():
    print(f"\nDataFrame: {name}")
    pd.to_datetime(df.index)
    display(df.iloc[:5, :5])  # Display first 5 rows and first 5 columns

index_name_list = data_frames3.keys()
print(index_name_list)

single_index_df = data_frames3['US (S&P 500)']

single_index_df.index = pd.to_datetime(single_index_df.index)
print(single_index_df.index)

"""### Visualize Original Data For a Single Index"""

import matplotlib.pyplot as plt

def plot_stock_data(df):
    """
    Plots OHLC data and Volume data from a stock DataFrame.

    Parameters:
    - df (pd.DataFrame): A DataFrame with columns ["Close", "High", "Low", "Open", "Volume"]
                         and datetime index.
    """
    fig, ax1 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot Close, High, Low, Open
    ax1[0].plot(df.index, df["Close"], label="Close")
    ax1[0].plot(df.index, df["High"], label="High")
    ax1[0].plot(df.index, df["Low"], label="Low")
    ax1[0].plot(df.index, df["Open"], label="Open")
    ax1[0].set_ylabel("Price")
    ax1[0].set_title("Stock Prices (OHLC)")
    ax1[0].legend()
    ax1[0].grid(True)

    # Plot Volume
    ax1[1].plot(df.index, df["Volume"], color='gray', label="Volume")
    ax1[1].set_ylabel("Volume")
    ax1[1].set_title("Trading Volume")
    ax1[1].legend()
    ax1[1].grid(True)

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

plot_stock_data(single_index_df)

import pandas as pd

def filter_close_by_year(df, year):
    """
    Filters the 'Close' column of a DataFrame for the specified year.

    Parameters:
    - df (pd.DataFrame): DataFrame with a DateTime index and a 'Close' column.
    - year (str): A string representing the year to filter by (e.g., '2018').

    Returns:
    - pd.Series: Filtered 'Close' values for the specified year.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DateTimeIndex.")

    try:
        year_int = int(year)
    except ValueError:
        raise ValueError(f"Invalid year: {year}. Must be a 4-digit number string.")

    return df[df.index.year == year_int]["Close"]

# Test the function
year = "2018"
filtered_close = filter_close_by_year(single_index_df, year)

# Plot the result
plt.figure(figsize=(10, 5))
plt.plot(filtered_close.index, filtered_close.values, color="green", marker='.', label=f"Close Prices in {year}")
plt.title(f"Close Prices in {year}")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

type(filtered_close.index)

def find_extrema_points(series, window_size):
    """
    Finds indices where the middle value of each window is equal to the max or min.

    Parameters:
    - series (pd.Series): Time series data with a DateTimeIndex.
    - window_size (int): Size of each sliding window.

    Returns:
    - (list, list): Tuple of two lists containing the indices of local maxima and minima.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The input series must have a DateTimeIndex.")
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")

    max_indices = []
    min_indices = []

    for i in range(0, len(series) - window_size + 1):
        window = series.iloc[i:i + window_size]
        mid_idx = i + window_size // 2  # choose first middle value if even
        mid_value = series.iloc[mid_idx]
        window_max = window.max()
        window_min = window.min()
        mid_timestamp = series.index[mid_idx]

        if mid_value == window_max:
            max_indices.append(mid_timestamp)
        elif mid_value == window_min:
            min_indices.append(mid_timestamp)

    return max_indices, min_indices

# Test the function
year = "2023"
close_df = filter_close_by_year(single_index_df, year)

close_df = single_index_df['Close']

window_size = 50
max_pts, min_pts = find_extrema_points(close_df, window_size)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(close_df.index, close_df.values, label="Apple Closing Price", color='blue')
plt.scatter(max_pts, close_df[max_pts], color='red', label='Local Max', zorder=5)
plt.scatter(min_pts, close_df[min_pts], color='green', label='Local Min', zorder=5)

title = f"Extrema Points (Window Size = {window_size})"
plt.title(title)
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(title, dpi=300, bbox_inches='tight')
plt.show()

"""## Compute Volatility Based on Daily Returns"""

import pandas as pd

# Create a DataFrame with two columns of integers from 1 to 100
df = pd.DataFrame({
    'A': range(1, 101),
    'B': range(1, 101)
})

print(df.head())

df['A'].rolling(window=4).sum()/4

folder = "2018_to_2024_by_companies"
data_frames3 = load_csvs_to_dict(folder)

single_index_df = data_frames3['US (S&P 500)']

single_index_df.index = pd.to_datetime(single_index_df.index)
print(single_index_df.index)

import numpy as np

def add_daily_return(df):
    """
    Adds a column 'return (n=1)' to the DataFrame which contains the daily log returns based on 'Close' prices.
    """
    df['return (n=1)'] = np.log(df['Close'] / df['Close'].shift(1))
    return df

def add_volatility(df, n):
    """
    Adds a column 'volatility (n={n})' to the DataFrame based on the rolling standard deviation
    of the daily returns over a window size of n.
    """
    if 'return (n=1)' not in df.columns:
        df = add_daily_return(df)  # Ensure daily return exists

    col_name = f"volatility (n={n})"
    df[col_name] = df['return (n=1)'].rolling(window=n).std()/(n**0.5)
    return df

# Test daily return
single_index_df = add_daily_return(single_index_df)

# Test volatility over 3-day window
single_index_df = add_volatility(single_index_df, n=20)

# To check the result
print(single_index_df)

single_index_df = single_index_df.dropna()
single_index_df.head()

single_index_df.shape

import matplotlib.pyplot as plt

# Plot daily return
plt.figure(figsize=(14, 5))
plt.plot(single_index_df.index, single_index_df['return (n=1)'], label='Daily Log Return', color='blue')
plt.title("Daily Log Return")
plt.xlabel("Date")
plt.ylabel("Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot rolling volatility (example: 20-day window)
plt.figure(figsize=(14, 5))
plt.plot(single_index_df.index, single_index_df['volatility (n=20)'], label='20-Day Rolling Volatility', color='orange')
plt.title("Rolling Volatility (20-Day)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

"""## Split Train and Test Data"""

def split_time_series(df, test_size=0.2, split_date=None):
    """
    Splits a time series DataFrame into train and test sets.

    Parameters:
    - df (pd.DataFrame): Time series DataFrame with date-based rows.
    - test_size (float): Fraction of data to use for testing (ignored if split_date is provided).
    - split_date (str or pd.Timestamp): Optional. Split point as a date (e.g., '2022-01-01').

    Returns:
    - train_df (pd.DataFrame): Training set
    - test_df (pd.DataFrame): Testing set
    """

    # Ensure index is sorted
    df = df.sort_index()

    if split_date is not None:
        train_df = df[df.index < split_date]
        test_df = df[df.index >= split_date]
    else:
        split_point = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_point]
        test_df = df.iloc[split_point:]

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

# Assume df has date index
single_index_df.index = pd.to_datetime(single_index_df.index)

# OR use a specific split date
train_df, test_df = split_time_series(single_index_df, split_date="2021-01-01")
print(train_df.shape, test_df.shape)

# 80/20 split
train_df, test_df = split_time_series(single_index_df, test_size=0.2)
print(train_df.shape, test_df.shape)

"""## Prepare Data for LSTM"""

min_max_1 = (single_index_df.min(), single_index_df.max())
min_max_1[0], min_max_1[1]

import numpy as np

def prepare_data_for_lstm(df, window_size, target, period, min_max=None):
    """
    Splits a DataFrame into overlapping windows of a given size.

    Parameters:
    - df (pd.DataFrame): Input DataFrame of shape (n, m)
    - window_size (int): Number of rows per window

    Returns:
    - result (list of np.ndarray): List of arrays, each of shape (window_size, m)
    """
    assert window_size > period, 'window_size need to be greater than perdiction period'
    print("Original DataFrame shape:", df.shape)

    input_list = []
    output_list = []

    col_min = 0
    col_range = 1
    if min_max is not None:
        col_min = min_max[0]
        col_max = min_max[1]
        col_range = col_max - col_min
    # print(col_range)
    # print(col_min)

    for i in range(len(df) - window_size + 1):
        input_rows = df.iloc[i:i + window_size - period]
        input_rows = (input_rows - col_min) / col_range
        target_row = df.iloc[i+window_size-1 : i+window_size]
        target_row = (target_row - col_min) / col_range

        target_val = target_row[target]

        input_rows = input_rows.to_numpy()
        input_list.append(input_rows)
        output_list.append(target_val)

    input_array = np.array(input_list)
    output_array = np.array(output_list)


    return input_array, output_array

# Dummy data
# df = pd.DataFrame(np.random.rand(10, 3), columns=['A', 'B', 'C'])

# Apply function with window size 4
lstm_input, lstm_output = prepare_data_for_lstm(single_index_df, window_size=40, target='volatility (n=20)', period=5)
print("Final result array shape:", lstm_input.shape, lstm_output.shape)

lstm_train_input, lstm_train_output = prepare_data_for_lstm(train_df, window_size=40, target='volatility (n=20)', period=1, min_max=min_max_1)
print("Final result array shape:", lstm_train_input.shape, lstm_train_output.shape)

lstm_test_input, lstm_test_output = prepare_data_for_lstm(test_df, window_size=40, target='volatility (n=20)', period=1, min_max=min_max_1)
print("Final result array shape:", lstm_test_input.shape, lstm_test_output.shape)

"""### Prepare Data for Regular Neural Network / Most other models"""

import numpy as np

def prepare_data_for_regular_model(df, window_size, target, period):
    """
    Splits a DataFrame into overlapping windows of a given size.

    Parameters:
    - df (pd.DataFrame): Input DataFrame of shape (n, m)
    - window_size (int): Number of rows per window

    Returns:
    - result (list of np.ndarray): List of arrays, each of shape (window_size, m)
    """
    assert window_size > period, 'window_size need to be greater than perdiction period'
    print("Original DataFrame shape:", df.shape)

    input_list = []
    output_list = []

    for i in range(len(df) - window_size + 1):
        input_rows = df.iloc[i:i + window_size - period]
        target_row = df.iloc[i+window_size-1 : i+window_size]

        target_val = target_row[target]

        input_rows = input_rows.to_numpy()
        flat_array = input_rows.flatten()
        input_list.append(flat_array)
        output_list.append(target_val)

    input_array = np.array(input_list)
    output_array = np.array(output_list)


    return input_array, output_array

# Dummy data
# df = pd.DataFrame(np.random.rand(10, 3), columns=['A', 'B', 'C'])

# Apply function with window size 4
regular_train_input, regular_train_output = prepare_data_for_regular_model(train_df, window_size=15, target='volatility (n=20)', period=1)
print("Final result array shape:", regular_train_input.shape, regular_train_output.shape)

regular_test_input, regular_test_output = prepare_data_for_regular_model(test_df, window_size=15, target='volatility (n=20)', period=1)
print("Final result array shape:", regular_test_input.shape, regular_test_output.shape)

"""## Build LSTM Model"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

lstm_model = Sequential([
    LSTM(units=128, return_sequences=True, input_shape=(35, 7)),  # First LSTM layer
    Dropout(0.2),  # Dropout to prevent overfitting
    LSTM(units=64, return_sequences=False),  # Second LSTM layer
    Dropout(0.2),
    Dense(units=25),  # Fully connected layer
    Dense(units=1)  # Output layer (predicting next day's spot price)
])

# Compile the model
lstm_model.compile(optimizer="adam", loss="mean_squared_error")

# Model Summary
lstm_model.summary()

# Train the model
lstm_history = lstm_model.fit(lstm_train_input, lstm_train_output, epochs=20, batch_size=16)

# Evaluate the model
test_loss = lstm_model.evaluate(lstm_test_input, lstm_test_output)
print(f"Test Loss: {test_loss:.4f}")

import matplotlib.pyplot as plt

plt.plot(lstm_history.history['loss'], label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("LSTM Training Performance")
plt.savefig("LSTM train history.png", dpi=300, bbox_inches='tight')
plt.show()

def visualize_predictions(model, X_train, y_train, X_test, y_test, df, title=""):
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Create a combined actual spot price series
    actual_spot_prices = np.concatenate([y_train, y_test])
    predicted_spot_prices = np.concatenate([y_train_pred.flatten(), y_test_pred.flatten()])

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(actual_spot_prices, color="black", label="Actual Value")
    plt.plot(predicted_spot_prices, color="red", linestyle="dotted", label="Predicted Value")
    # plt.plot(futures_prices_all, color="red", linestyle="dotted", label="Futures Price")

    # Mark the transition between training and test sets
    plt.axvline(len(y_train), color="gray", linestyle="--", label="Train-Test Split")
    plt.title("Actual vs. Predicted")
    plt.legend()
    plt.savefig(title, dpi=300, bbox_inches='tight')
    plt.show()

visualize_predictions(lstm_model, lstm_train_input, lstm_train_output, lstm_test_input, lstm_test_output, single_index_df, "lstm_prediction")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test).flatten()

        # Compute metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Store results
        results.append({"Model": name, "MSE": mse, "RMSE": rmse, "R2": r2})

    return pd.DataFrame(results)

models2 = {
    "LSTM": lstm_model
}

results_dl = evaluate_models(models2, lstm_test_input, lstm_test_output)  # Deep learning models (RNN, LSTM, GRU)

display(results_dl)
