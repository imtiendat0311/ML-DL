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
x_total = np.concatenate([x_train, x_test], axis=0)

param_grid = {'hidden_layers': [2, 3, 4], 'hidden_units': [32, 64, 128, 256]}


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_model(data, hidden_layers, hidden_units, encoded):
    model = {}
    # create first hidden layer
    model['W1'] = np.random.randn(
        data.shape[1], hidden_units[0]) / np.sqrt(x_train.shape[1])
    model['b1'] = np.zeros((1, hidden_units[0]))
    # create hidden layers
    for i in range(hidden_layers - 1):
        model['W'+str(i+2)] = np.random.randn(hidden_units[i],
                                              hidden_units[i+1]) / np.sqrt(hidden_units[i])
        model['b'+str(i+2)] = np.zeros((1, hidden_units[i+1]))
    # create output layer
    model['W' + str(hidden_layers+1)] = np.random.randn(hidden_units[-1],
                                                        x_train.shape[1]) / np.sqrt(hidden_units[-1])
    model['b' + str(hidden_layers+1)] = np.zeros((1, x_train.shape[1]))
    model['encoded_layer'] = encoded
    model['decoded_layer'] = (hidden_layers+1) - encoded
    return model


def encoder(model, batch):
    for i in range(model['encoded_layer']):
        if (i == 0):
            z = np.dot(batch, model['W1']) + model['b1']
            a = sigmoid(z)
        else:
            z = np.dot(model['a'+str(i)], model['W'+str(i+1)]
                       ) + model['b'+str(i+1)]
            a = sigmoid(z)
        model['z'+str(i+1)] = z
        model['a'+str(i+1)] = a


def decoder(model):
    for i in range(model['decoded_layer']):
        z = np.dot(model['a'+str(model['encoded_layer']+i)], model['W'+str(model['encoded_layer']+i+1)]
                   ) + model['b'+str(model['encoded_layer']+i+1)]
        a = sigmoid(z)
        model['z'+str(model['encoded_layer']+1+i)] = z
        model['a'+str(model['encoded_layer']+1+i)] = a


def train(model, training_set, learning_rate=0.0001):
    num_epochs = 11
    batch_size = 256
    num_batches = training_set.shape[0] // batch_size
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in range(num_batches):
            batch_indices = np.random.choice(training_set.shape[0], batch_size)
            x_batch = x_train[batch_indices]  # batch data
            # forward propagation
            encoder(model, x_batch)  # encoded process
            decoder(model)  # decoded process
            # mean square error
            loss = np.mean(
                (model['a'+str(model['encoded_layer']+model['decoded_layer'])] - x_batch)**2)
            total_loss += loss  # total loss
            dL_dz = []
            # backpropagation
            for i in reversed(range(1, model['encoded_layer']+model['decoded_layer']+1)):
                if (i == model['encoded_layer']+model['decoded_layer']):
                    dL_da = model['a'+str(i)] - x_batch
                else:
                    dL_da = np.dot(dL_dz[-1], model['W'+str(i+1)].T)
                    # sigmoid derivative
                dL_dz.append(dL_da * model['a'+str(i)]
                             * (1 - model['a'+str(i)]))
                # derivative respect to weight and bias
                dL_dW = np.dot(model['a'+str(i-1)].T, dL_dz[-1]) if i - \
                    1 > 0 else np.dot(x_batch.T, dL_dz[-1])
                dL_db = np.sum(dL_dz[-1], axis=0, keepdims=True)
                # update weight and bias
                model['W'+str(i)] -= learning_rate * dL_dW
                model['b'+str(i)] -= learning_rate * dL_db
        if (epoch % 10 == 0):
            print("Epoch {0}: {1}".format(epoch, total_loss / num_batches))


best_loss = np.inf
best_model = None
k = 5  # k fold
for hidden_layer in param_grid['hidden_layers']:
    hidden_unit = np.array(param_grid['hidden_units'][0:hidden_layer])
    np.random.shuffle(hidden_unit)
    avg = 0
    for i in range(k):
        test_set = x_total[i*x_total.shape[0]//k:(i+1)*x_total.shape[0]//k]
        train_set = np.concatenate(
            [x_total[:i*x_total.shape[0]//k], x_total[(i+1)*x_total.shape[0]//k:]])
        model = create_model(train_set, hidden_layer,
                             hidden_unit, hidden_layer//2)
        train(model, train_set)
        encoder(model, test_set)
        decoder(model)
        avg += np.mean((model['a'+str(hidden_layer+1)] - test_set)**2)
    avg /= k
    print('Hidden Layers: {0}, Hidden Units: {1}, Loss: {2}'.format(
        hidden_layer, hidden_unit, avg))
    if (avg < best_loss):
        best_loss = avg
        best_model = {'hidden_layers': hidden_layer,
                      'hidden_units': hidden_unit}
print('Best Loss: {0}'.format(best_loss))
print('Best Model: Hidden layers : {0}, Hidden Units : {1}'.format(
    best_model['hidden_layers'], best_model['hidden_units']))
