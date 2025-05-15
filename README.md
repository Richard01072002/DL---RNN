# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## THEORY
Recurrent Neural Networks (RNNs) are designed to work with sequential data by maintaining a "memory" of previous inputs using hidden states. This makes them ideal for tasks like time series prediction. The model takes a sequence of past stock prices and learns to predict the next price.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import the required libraries and load the stock price dataset.

### STEP 2: 

Normalize the closing price values using MinMaxScaler.

### STEP 3: 

Create input sequences and output labels for the RNN using a sliding window approach.

### STEP 4: 

Define an RNN model using PyTorch's nn.Module with two layers and a hidden size of 64.

### STEP 5: 

Train the RNN using Mean Squared Error (MSE) loss and Adam optimizer over multiple epochs.

### STEP 6: 

Evaluate the model using the test data and visualize predictions against actual stock prices.



## PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load and Preprocess Data
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the Model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    print("by RICHARDSON A")
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs \n BY RICHARDSON')
    plt.legend()
    plt.show()

train_model(model, train_loader, criterion, optimizer)

# Predict on Test Data
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN \n by RICHARDSON A')
plt.legend()
plt.show()

print("by RICHARDSON A")
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')

```
### Name:

### Register Number:

```python
# Define RNN Model
class RNNModel(nn.Module):
    # write your code here




# Train the Model

# Write your code here


```

### OUTPUT

## Training Loss Over Epochs Plot
<img width="693" alt="Screenshot 2025-05-15 at 2 52 09 PM" src="https://github.com/user-attachments/assets/f93b4f1e-7e3d-432f-b716-b81714ccd9dd" />


## True Stock Price, Predicted Stock Price vs time
<img width="991" alt="Screenshot 2025-05-15 at 2 52 59 PM" src="https://github.com/user-attachments/assets/3e859beb-b4a3-46b1-be5b-4c42a1a6991b" />

<img width="994" alt="Screenshot 2025-05-15 at 2 53 46 PM" src="https://github.com/user-attachments/assets/4edc7757-6cb9-4abc-82b5-7cc7fa283011" />

### Predictions
Predicted Price: [173.5937]
Actual Price: [172.8415]

## RESULT
The RNN model was successfully developed and trained to predict stock prices. The model learned from the historical closing price data and was able to predict the future price with a close approximation to the actual value.
