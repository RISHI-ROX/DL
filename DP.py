Program1
Build a deep neural network model start with linear regression using a single variable
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
# Step 1: Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1) # Single input feature
y = 3.5 * X + np.random.randn(100, 1) * 2 # Linear relation with noise
# Step 2: Define the model
model = Sequential([
Dense(1, input_dim=1, activation='linear') # Single input and single output
])
# Step 3: Compile the model
model.compile(optimizer=SGD(learning_rate=0.01), loss='mse', metrics=['mae'])
# Step 4: Train the model
history = model.fit(X, y, epochs=100, batch_size=10, verbose=1)
# Step 5: Evaluate the model
loss, mae = model.evaluate(X, y, verbose=0)
print(f"Final Loss: {loss:.4f}, Mean Absolute Error: {mae:.4f}")
# Step 6: Visualize the results
y_pred = model.predict(X)
plt.scatter(X, y, label='True Data', color='blue')
plt.plot(X, y_pred, label='Model Prediction', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression with Neural Network')
plt.show()



Program2
import numpy as np import pandas as pd import matplotlib.pyplot as plt import seaborn as sns from tensorflow.keras.models import Sequential from tensorflow.keras.layers import Dense from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler 
 
# Create synthetic data np.random.seed(0) 
X1 = np.random.rand(100) * 10  # Feature 1 X2 = np.random.rand(100) * 20  # Feature 2 
y = 3 * X1 + 2 * X2 + np.random.randn(100) * 5  # Target variable with noise 
 
# Combine features into a DataFrame 
data = pd.DataFrame({'Feature1': X1, 'Feature2': X2, 'Target': y}) 
X = data[['Feature1', 'Feature2']] 
y = data['Target'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
 
# Define the model model = Sequential() model.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))  # Hidden layer model.add(Dense(32, activation='relu'))   # Second hidden layer model.add(Dense(1))   # Output layer 
 
# Compile the model 
model.compile(optimizer='adam', loss='mean_squared_error') model.fit(X_train_scaled, y_train, epochs=100, batch_size=10) y_pred = model.predict(X_test_scaled) 
plt.figure(figsize=(12, 5)) 
plt.figure(figsize=(10, 6)) 
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual') plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Ideal Line') plt.title('Predicted vs Actual Values') plt.xlabel('Actual Values') plt.ylabel('Predicted Values') plt.legend() plt.grid() 
plt.show() 
 



Program3
import numpy as np from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
 
# Define the truth table for a logic gate (e.g., XOR) # Replace 'gate' with AND, OR, XOR as needed. def get_data(gate="XOR"): 
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])     if gate == "AND": 
        y = np.array([[0], [0], [0], [1]])  # AND Gate     elif gate == "OR": 
        y = np.array([[0], [1], [1], [1]])  # OR Gate     elif gate == "XOR": 
        y = np.array([[0], [1], [1], [0]])  # XOR Gate     else: 
        raise ValueError("Supported gates are AND, OR, XOR")     return X, y 
 
# Choose the gate to predict 
logic_gate = "XOR"  # Change this to "AND" or "OR" for other gates 
X, y = get_data(logic_gate) 
 
# Build the feedforward neural network 
model = Sequential([ 
    Dense(4, input_dim=2, activation='relu'),  # Hidden layer with 4 neurons 
    Dense(1, activation='sigmoid')            # Output layer with 1 neuron 
]) 
 
# Compile the model 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
 
# Train the model print(f"Training model for {logic_gate} gate...") 
model.fit(X, y, epochs=100, verbose=1) 
 
predictions = model.predict(X) for i in range(len(X)): 
print(f"Input: {X[i]}, Predicted Output: {round(predictions[i][0])}, Actual Output: {y[i][0]}") 



Program 4
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout from tensorflow.keras.utils import to_categorical from tensorflow.keras.datasets import mnist import numpy as np 
import matplotlib.pyplot as plt 
 
# Load the MNIST dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
 
# Preprocess the data 
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0  # Normalize and add channel dimension x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0 
 
# One-hot encode labels y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10) 
 
# Build the CNN model model = Sequential([ 
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
    MaxPooling2D(pool_size=(2, 2)), 
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D(pool_size=(2, 2)), 
    Flatten(), 
    Dense(128, activation='relu'), 
    Dropout(0.5), 
    Dense(10, activation='softmax')  # Output layer for 10 classes (digits 0-9) 
]) 
 
# Compile the model 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
 
# Train the model 
history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=128) 
 
# Evaluate the model 
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2) print(f"Test Accuracy: {test_accuracy:.4f}") 
# Save the model for future use
model.save("character_recognition_cnn.h5")
# Load and test the model (optional)
# loaded_model = tf.keras.models.load_model("character_recognition_cnn.h5")
# predictions = loaded_model.predict(x_test)
# Visualize a few predictions
def visualize_predictions(data, labels, model, num_samples=5):
for i in range(num_samples):
plt.imshow(data[i].reshape(28, 28), cmap="gray")
plt.title(f"Prediction: {np.argmax(model.predict(data[i:i+1]))}, Label: {np.argmax(labels[i])}")
plt.axis("off")
plt.show()
visualize_predictions(x_test, y_test, model)




Program 5:

import numpy as np import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import SimpleRNN, Dense, Flatten, Conv2D, MaxPooling2D, LSTM from tensorflow.keras.datasets import mnist from tensorflow.keras.utils import to_categorical 
 
# Load the MNIST dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
 
# Preprocess data 
x_train = x_train / 255.0  # Normalize pixel values x_test = x_test / 255.0 
y_train = to_categorical(y_train, 10)  # One-hot encode labels y_test = to_categorical(y_test, 10) 
 
# Reshape for RNN input: (samples, time_steps, features) x_train_rnn = x_train.reshape(-1, 28, 28) 
x_test_rnn = x_test.reshape(-1, 28, 28) 
 
# Reshape for CNN input: (samples, height, width, channels) x_train_cnn = x_train.reshape(-1, 28, 28, 1) x_test_cnn = x_test.reshape(-1, 28, 28, 1) 
 
# RNN Model 
rnn_model = Sequential([ 
    SimpleRNN(128, input_shape=(28, 28), activation='relu'), 
    Dense(10, activation='softmax') 
]) 
 
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
 
# CNN Model 
cnn_model = Sequential([ 
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
    MaxPooling2D((2, 2)), 
    Flatten(), 
    Dense(128, activation='relu'), 
    Dense(10, activation='softmax') 
]) 
 
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train and evaluate RNN model
print("Training RNN...")
rnn_model.fit(x_train_rnn, y_train, epochs=5, batch_size=128, validation_data=(x_test_rnn, y_test))
rnn_loss, rnn_accuracy = rnn_model.evaluate(x_test_rnn, y_test)
# Train and evaluate CNN model
print("\nTraining CNN...")
cnn_model.fit(x_train_cnn, y_train, epochs=5, batch_size=128, validation_data=(x_test_cnn, y_test))
cnn_loss, cnn_accuracy = cnn_model.evaluate(x_test_cnn, y_test)
# Print results
print("\nRNN Accuracy: {:.2f}%".format(rnn_accuracy * 100))
print("CNN Accuracy: {:.2f}%".format(cnn_accuracy * 100))