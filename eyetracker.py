#import libraries
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Define the name of the zip file and the folder where the data files will be extracted
DATA_ZIP = "EyeT.zip"
DATA_DIR = "EyeT"

# Preprocessing the data
# Define a function to preprocess the eye tracker data
def preprocess(data_dir):
    # Get a list of all CSV files in the directory
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
    # Read the eyetracker data from files
    data_files = ['data1.csv', 'data2.csv', 'data3.csv']
    df_list = []
    for file in data_files:
        with open(file, 'r') as f:
          df_list.append(pd.read_csv(f))
    df = pd.concat(df_list)

   # Close the files
    for file in df_list:
     file.close()

    # Read the data from all CSV files into a single DataFrame
    df = pd.concat([pd.read_csv(f) for f in csv_files])

    # Calculate the saccades, pupil dilation, blink rate, gaze duration, and fixation
    # Columns in the data are: timestamp, x, y, pupil_dilation, saccade, blink, fixation
    df['duration'] = df.groupby('fixation')['timestamp'].diff().fillna(0)
    df['fixation_count'] = df.groupby('fixation')['timestamp'].transform('count')
    df['blink_rate'] = df['blink'] / df['duration']
    df['gaze_duration'] = df['duration'] / df['fixation_count']
    df = df[['pupil_dilation', 'saccade', 'blink_rate', 'gaze_duration', 'fixation']]
    
    return df


# Create the empathy measurement model
# The target variable is 'empathy', and it is based on the other features
X = df.drop('empathy', axis=1)
y = df['empathy']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Define a function to train the empathy model
def train(df):
    # Extract the relevant features from the DataFrame
    X = df[["saccades", "pupil_dilation", "blink_rate", "gaze_duration", "fixation"]]
    # Extract the empathy score from the DataFrame
    y = df["empathy_score"]

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    return model

# Function to evaluate the empathy model
def evaluate(model, test_data):
    # Extract relevant features from the test data
    X_test = test_data[["saccades", "pupil_dilation", "blink_rate", "gaze_duration", "fixation"]]
    # Extract the true empathy score from the EyeT data
    y_test = test_data["empathy_score"]

    # Use the trained model to predict the empathy score for the test data
    y_pred = model.predict(X_test)

    # Calculate the mean squared error of the predictions
    mse = ((y_pred - y_test) ** 2).mean()

    return mse

if __name__ == "__main__":
    # Extract the data files from the zip archive
    with zipfile.ZipFile(DATA_ZIP, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    # Preprocess the data
    df = preprocess(DATA_DIR)

    # Split the data into training and test sets
    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)

    # Train the model
    model = train(train_data)

    # Evaluate the model on the test data
    mse = evaluate(model, test_data)

    # Print the mean squared error of the predictions
    print(f"Mean squared error: {mse}")

    # Remove the extracted data directory
    os.remove(DATA_ZIP)
    os.rmdir(DATA_DIR)
