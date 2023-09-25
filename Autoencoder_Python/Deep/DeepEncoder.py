import numpy as np
from mnist import MNIST

mndata = MNIST(
    '/Users/boo/Desktop/coding/Deep_Learning/MNIST')
mndata.gz = True
x_train, _ = mndata.load_training()
x_test, _ = mndata.load_testing()
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

input_dim = x_train.shape[1]
hidden_dim1 = 32
hidden_dim2 = 64
hidden_dim3 = 128
output_dim = input_dim

# Initialize the weight matrices and biases
W1 = np.random.randn(input_dim, hidden_dim1) * 0.01
b1 = np.zeros((1, hidden_dim1))
W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.01
b2 = np.zeros((1, hidden_dim2))
W3 = np.random.randn(hidden_dim2, hidden_dim3) * 0.01
b3 = np.zeros((1, hidden_dim3))
W4 = np.random.randn(hidden_dim3, output_dim) * 0.01
b4 = np.zeros((1, output_dim))

# Define the activation function (ReLU) and the sigmoid function


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def encoder(x):
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = relu(z3)
    return a3, a2, a1, z3, z2, z1


def decoder(x):
    z4 = np.dot(x, W4) + b4
    a4 = sigmoid(z4)
    return a4


# Define the training loop
num_epochs = 100
batch_size = 256
num_batches = x_train.shape[0] // batch_size
learning_rate = 0.0001

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in range(num_batches):
        # Select a random batch of images
        batch_indices = np.random.choice(x_train.shape[0], batch_size)
        x_batch = x_train[batch_indices]

        # Forward pass through the encoder and decoder
        a3, a2, a1, z3, z2, z1 = encoder(x_batch)
        x_hat = decoder(a3)

        # Compute the reconstruction loss and the gradients
        loss = np.mean((x_hat - x_batch)**2)
        total_loss += loss
        dL_dx_hat = x_hat - x_batch
        dL_dz4 = dL_dx_hat * x_hat * (1 - x_hat)
        dL_dW4 = np.dot(a3.T, dL_dz4)
        dL_db4 = np.sum(dL_dz4, axis=0, keepdims=True)

        dL_da3 = np.dot(dL_dz4, W4.T)
        dL_dz3 = dL_da3 * (z3 > 0)
        dL_dW3 = np.dot(a2.T, dL_dz3)
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)

        dL_da2 = np.dot(dL_dz3, W3.T)
        dL_dz2 = dL_da2 * (z2 > 0)
        dL_dW2 = np.dot(a1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = np.dot(dL_dz2, W2.T)
        dL_dz1 = dL_da1 * (z1 > 0)
        dL_dW1 = np.dot(x_batch.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Update the weights and biases
        W1 -= learning_rate * dL_dW1
        b1 -= learning_rate * dL_db1
        W2 -= learning_rate * dL_dW2
        b2 -= learning_rate * dL_db2
        W3 -= learning_rate * dL_dW3
        b3 -= learning_rate * dL_db3
        W4 -= learning_rate * dL_dW4
        b4 -= learning_rate * dL_db4

    # Print the reconstruction loss every 10 epochs
    if epoch % 10 == 0:
        print("Epoch %d: loss = %f" % (epoch, total_loss))


# Use the encoder to obtain compressed representations of new images
encoded_imgs, _, _, _, _, _ = encoder(x_test)

# Use the decoder to reconstruct images from the compressed representations
decoded_imgs = decoder(encoded_imgs)

print("Done")
print("Loss = %f" % np.mean((x_test - decoded_imgs)**2))
