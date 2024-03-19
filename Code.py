import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

# Create DataFrame from provided data
data = {
    'OrderDate': ['1-6-18', '1-23-18', '2-9-18', '2-26-18', '3-15-18', '4-1-18', '4-18-18', '5-5-18', '5-22-18',
                  '6-8-18', '6-25-18', '7-12-18', '7-29-18', '8-15-18', '9-1-18', '9-18-18', '10-5-18', '10-22-18',
                  '11-8-18', '11-25-18', '12-12-18', '12-29-18', '1-15-19', '2-1-19', '2-18-19', '3-7-19', '3-24-19',
                  '4-10-19', '4-27-19', '5-14-19', '5-31-19', '6-17-19', '7-4-19', '7-21-19', '8-7-19', '8-24-19',
                  '9-10-19', '9-27-19', '10-14-19', '10-31-19', '11-17-19', '12-4-19', '12-21-19'],
    'Region': ['East', 'Central', 'Central', 'Central', 'West', 'East', 'Central', 'Central', 'West', 'East',
               'Central', 'East', 'East', 'East', 'Central', 'East', 'Central', 'East', 'East', 'Central', 'Central',
               'East', 'Central', 'East', 'West', 'Central', 'Central', 'East', 'Central', 'West', 'Central', 'East',
               'East', 'Central', 'Central', 'Central', 'West', 'Central', 'West', 'Central', 'Central', 'Central',
               'Central', 'Central'],
    'Manager': ['Martha', 'Hermann', 'Hermann', 'Timothy', 'Timothy', 'Martha', 'Martha', 'Hermann', 'Douglas',
                'Martha', 'Hermann', 'Martha', 'Douglas', 'Martha', 'Douglas', 'Martha', 'Hermann', 'Martha', 'Douglas',
                'Hermann', 'Douglas', 'Martha', 'Douglas', 'Martha', 'West', 'Martha', 'Hermann', 'Martha', 'Douglas',
                'Hermann', 'Martha', 'Hermann', 'Hermann', 'Martha', 'Douglas', 'Martha', 'Hermann', 'Martha', 'Douglas',
                'Hermann', 'Hermann', 'Martha', 'Martha', 'Martha'],
    'SalesMan': ['Alexander', 'Shelli', 'Luis', 'David', 'Stephen', 'Alexander', 'Steven', 'Luis', 'Michael', 'Alexander',
                 'Sigal', 'Diana', 'Karen', 'Alexander', 'John', 'Alexander', 'Sigal', 'Alexander', 'Karen', 'Shelli',
                 'John', 'Alexander', 'John', 'Alexander', 'Stephen', 'David', 'Michael', 'David', 'John', 'Sigal',
                 'David', 'Alexander', 'Sigal', 'Shelli', 'Alexander', 'David', 'Alexander', 'Shelli', 'Stephen', 'David',
                 'Stephen', 'Michael', 'Steven', 'Luis', 'Luis', 'Steven'],
    'Item': ['Television', 'Home Theater', 'Television', 'Cell Phone', 'Television', 'Home Theater', 'Television', 'Television',
             'Television', 'Home Theater', 'Television', 'Home Theater', 'Home Theater', 'Television', 'Desk', 'Video Games',
             'Home Theater', 'Cell Phone', 'Cell Phone', 'Video Games', 'Television', 'Video Games', 'Home Theater', 'Home Theater',
             'Desk', 'Home Theater', 'Video Games', 'Television', 'Home Theater', 'Television', 'Video Games', 'Video Games',
             'Video Games', 'Desk', 'Television', 'Cell Phone', 'Home Theater', 'Television', 'Video Games', 'Television',
             'Cell Phone', 'Home Theater', 'Home Theater', 'Home Theater', 'Home Theater', 'Home Theater'],
    'Units': [95, 50, 36, 27, 56, 60, 75, 90, 32, 60, 90, 29, 81, 35, 2, 16, 28, 64, 15, 96, 67, 74, 46, 87, 4, 7, 50,
              66, 96, 53, 80, 5, 62, 55, 42, 3, 7, 76, 57, 14, 11, 94, 28],
    'Unit_price': [1198.00, 500, 1198.00, 225, 1198.00, 500, 1198.00, 1198.00, 1198.00, 500, 1198.00, 500, 500, 1198.00,
                   125, 58.5, 500, 225, 225, 58.5, 1198.00, 58.5, 500, 500, 125, 500, 58.5, 1198.00, 500, 1198.00, 58.5,
                   58.5, 58.5, 125, 1198.00, 225, 500, 1198.00, 58.5, 1198.00, 225, 500, 500, 500, 500, 500],
    'Sale_amt': [113810.00, 25000.00, 43128.00, 6075.00, 67088.00, 30000.00, 89850.00, 107820.00, 38336.00, 30000.00,
                 107820.00, 14500.00, 40500.00, 41930.00, 250, 936, 14000.00, 14400.00, 3375.00, 5616.00, 80266.00, 4329.00,
                 23000.00, 43500.00, 2000.00, 3500.00, 2925.00, 79068.00, 21600.00, 63494.00, 40000.00, 625, 3627.00, 3217.50, 2457.00, 375, 8386.00, 17100.00,
                 28500.00, 16772.00, 5500.00, 47000.00, 14000.00]
}

df = pd.DataFrame(data)

# Convert 'OrderDate' column to datetime
df['OrderDate'] = pd.to_datetime(df['OrderDate'], format='%m-%d-%y')

# Extracting Year, Month, and Day features
df['Year'] = df['OrderDate'].dt.year
df['Month'] = df['OrderDate'].dt.month
df['Day'] = df['OrderDate'].dt.day

# Dropping 'OrderDate' column as we have extracted features from it
df.drop(columns=['OrderDate'], inplace=True)

# One-hot encoding categorical variables 'Region', 'Manager', and 'SalesMan'
df = pd.get_dummies(df, columns=['Region', 'Manager', 'SalesMan'])

# Splitting data into features (X) and target (y)
X = df.drop(columns=['Sale_amt'])
y = df['Sale_amt']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Making predictions
y_pred = model.predict(X_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Save the model to a pickle file
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

