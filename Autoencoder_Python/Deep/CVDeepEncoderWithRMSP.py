import numpy as np
from mnist import MNIST
import itertools
mndata = MNIST(
    '/Users/boo/Desktop/coding/Deep_Learning/MNIST')
mndata.gz = True
x_train, _ = mndata.load_training()
x_test, _ = mndata.load_testing()
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

param_grid = {'hidden_layers': [2, 3, 4], 'hidden_units': [32, 64, 128, 256]}


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_model(data, hidden_layers, hidden_units, encoded):
    model = {}
    # create first hidden layer
    model['W1'] = np.random.randn(
        data.shape[1], hidden_units[0]) / np.sqrt(data.shape[1])
    model['b1'] = np.zeros((1, hidden_units[0]))
    # model['v_W1'] = np.zeros((data.shape[1], hidden_units[0]))
    # model['v_b1'] = np.zeros((1, hidden_units[0]))
    # create hidden layers
    for i in range(hidden_layers - 1):
        model['W'+str(i+2)] = np.random.randn(hidden_units[i],
                                              hidden_units[i+1]) / np.sqrt(hidden_units[i])
        model['b'+str(i+2)] = np.zeros((1, hidden_units[i+1]))
        # model['v_W'+str(i+2)] = np.zeros((hidden_units[i], hidden_units[i+1]))
        # model['v_b'+str(i+2)] = np.zeros((1, hidden_units[i+1]))
    # create output layer
    model['W' + str(hidden_layers+1)] = np.random.randn(hidden_units[-1],
                                                        data.shape[1]) / np.sqrt(hidden_units[-1])
    model['b' + str(hidden_layers+1)] = np.zeros((1, data.shape[1]))
    # model['v_W' + str(hidden_layers+1)] = np.zeros(
    #     (hidden_units[-1], data.shape[1]))
    # model['v_b' + str(hidden_layers+1)] = np.zeros((1, data.shape[1]))
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


def train(model, training_set, learning_rate=0.001, beta=0.9, epsilon=1e-8):
    num_epochs = 2
    batch_size = 256
    num_batches = training_set.shape[0] // batch_size
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in range(num_batches):
            batch_indices = np.random.choice(training_set.shape[0], batch_size)
            x_batch = training_set[batch_indices]  # batch data
            # forward propagation
            encoder(model, x_batch)  # encoded process
            decoder(model)  # decoded process
            # mean square error
            loss = np.mean(
                (model['a'+str(model['encoded_layer']+model['decoded_layer'])] - x_batch)**2)
            total_loss += loss  # total loss
            dL_dz = []
            # backpropagation Mini Batch
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

                # # momentum of W and b
                # model['v_W'+str(i)] = beta * model['v_W' +
                #                                    str(i)] + (1-beta)*dL_dW**2
                # model['v_b'+str(i)] = beta * model['v_b' +
                #                                    str(i)]+(1-beta)*dL_db**2
                # # update weight and bias
                # model['W'+str(i)] -= learning_rate * dL_dW / \
                #     (np.sqrt(model['v_W'+str(i)])+epsilon)
                # model['b'+str(i)] -= learning_rate * dL_db / \
                #     (np.sqrt(model['v_b'+str(i)])+epsilon)


best_loss = np.inf
best_model = None
best_model_string = None
k = 5  # k fold
count = 0
for hidden_layer in param_grid['hidden_layers']:
    # hidden_unit = np.array(param_grid['hidden_units'][0:hidden_layer])
    avg = 0
    for hidden_unit in itertools.product(param_grid['hidden_units'], repeat=hidden_layer):
        count += 1
        for i in range(k):
            test_set = x_train[i*x_train.shape[0]//k:(i+1)*x_train.shape[0]//k]
            train_set = np.concatenate(
                [x_train[:i*x_train.shape[0]//k], x_train[(i+1)*x_train.shape[0]//k:]])
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
            best_model_string = {'hidden_layers': hidden_layer,
                                 'hidden_units': hidden_unit}
            best_model = model
print('Best Loss: {0} out of {1} combination'.format(best_loss, count))
print('Best Model: Hidden layers : {0}, Hidden Units : {1}'.format(
    best_model['hidden_layers'], best_model['hidden_units']))

encoder(best_model, x_test)
decoder(best_model)
print("Loss of best model on test set = %f" % np.mean(
    (x_test - best_model['a'+str(best_model['encoded_layer']+best_model['decoded_layer'])])**2))


# number of k function pointer run cross validation
