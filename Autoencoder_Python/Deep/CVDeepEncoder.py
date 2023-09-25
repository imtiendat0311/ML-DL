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


def create_model(hidden_layers, hidden_units):
    model = {}
    # create first hidden layer
    model['W1'] = np.random.randn(
        x_train.shape[1], hidden_units[0]) / np.sqrt(x_train.shape[1])
    model['b1'] = np.zeros((1, hidden_units[0]))
    for i in range(hidden_layers - 1):
        model['W'+str(i+2)] = np.random.randn(hidden_units[i],
                                              hidden_units[i+1]) / np.sqrt(hidden_units[i])
        model['b'+str(i+2)] = np.zeros((1, hidden_units[i+1]))

    # create output layer
    model['W' + str(hidden_layers+1)] = np.random.randn(hidden_units[-1],
                                                        x_train.shape[1]) / np.sqrt(hidden_units[-1])
    model['b' + str(hidden_layers+1)] = np.zeros((1, x_train.shape[1]))
    return model


def encoder(model, hidden_layers, batch):
    encoded = {
        'z': [],
        'a': []
    }

    for i in range(hidden_layers//2):
        if (i == 0):
            z1 = np.dot(batch, model['W1']) + model['b1']
            a1 = sigmoid(z1)
            encoded['z'].append(z1)
            encoded['a'].append(a1)
        else:
            z = np.dot(encoded['a'][i-1], model['W'+str(i+1)]
                       ) + model['b'+str(i+1)]
            a = sigmoid(z)
            encoded['z'].append(z)
            encoded['a'].append(a)
    return encoded


def decoder(encoded, model, hidden_layers):
    decoder = {
        'z': [],
        'a': []
    }
    length = hidden_layers+1 - hidden_layers//2
    for i in range(length):
        if (i == 0):
            z = np.dot(encoded['a'][-1], model['W'+str(hidden_layers//2+1)]) + \
                model['b'+str(hidden_layers//2+1)]
            a = sigmoid(z)
            decoder['z'].append(z)
            decoder['a'].append(a)
        else:
            z = np.dot(decoder['a'][-1], model['W'+str(hidden_layers//2+1+i)]) + \
                model['b'+str(hidden_layers//2+1+i)]
            a = sigmoid(z)
            decoder['z'].append(z)
            decoder['a'].append(a)
    return decoder


def train(model, hidden_layers, training_set, learning_rate=0.001):
    num_epochs = 11
    batch_size = 256
    num_batches = training_set.shape[0] // batch_size

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in range(num_batches):
            batch_indices = np.random.choice(training_set.shape[0], batch_size)
            x_batch = x_train[batch_indices]

            encoded = encoder(model, hidden_layers, x_batch)
            decoded = decoder(encoded, model, hidden_layers)

            loss = np.mean((x_batch-decoded['a'][-1])**2)

            total_loss += loss
            dL_da = []
            dL_dz = []
            dL_W = []
            dL_b = []

            for i in range(hidden_layers+1):
                current_layer = hidden_layers+1-i

                # calculate dL_da
                if (current_layer == hidden_layers+1):
                    dL_da.append(decoded['a'][-1] - x_batch)
                else:
                    dL_da.append(
                        np.dot(dL_dz[-1], model['W'+str(current_layer+1)].T))

                # calculate dL_dz
                if (current_layer > hidden_layers//2):
                    dL_dz.append(dL_da[-1]*decoded['a'][-1-i]
                                 * (1 - decoded['a'][-1-i]))
                    # dL_dz.append(
                    #     dL_da[-1]*np.where(decoded['z'][-1-i] > 0, 1, 0))
                else:
                    dL_dz.append(dL_da[-1]*encoded['a'][current_layer-1]
                                 * (1 - encoded['a'][current_layer-1]))
                    # dL_dz.append(
                    #     dL_da[-1]*np.where(encoded['z'][current_layer-1] > 0, 1, 0))
                # calculate dL_dW
                if (current_layer > hidden_layers//2+1):
                    dL_W.append(np.dot(decoded['a'][-2-i].T, dL_dz[-1]))
                elif (i == hidden_layers):
                    dL_W.append(np.dot(x_batch.T, dL_dz[-1]))
                else:
                    dL_W.append(
                        np.dot(encoded['a'][current_layer-2].T, dL_dz[-1]))
                # calculate dL_db
                dL_b.append(np.sum(dL_dz[-1], axis=0, keepdims=True))

            for i in range(len(dL_W)):
                model['W'+str(i+1)] -= learning_rate * dL_W[-1-i]
                model['b'+str(i+1)] -= learning_rate * dL_b[-1-i]
        if (epoch % 10 == 0):
            print('Epoch {0}: Loss = {1}'.format(
                epoch, total_loss / num_batches))


best_loss = np.inf
k = 5  # k fold
best_param = {}
for hidden_layer in param_grid['hidden_layers']:
    hidden_unit = param_grid['hidden_units'][0:hidden_layer]
    avg = 0
    for i in range(k):
        model = create_model(hidden_layer, hidden_unit)
        test_set = x_total[i*x_total.shape[0]//k:(i+1)*x_total.shape[0]//k]
        train_set = np.concatenate(
            [x_total[:i*x_total.shape[0]//k], x_total[(i+1)*x_total.shape[0]//k:]])
        train(model, hidden_layer, train_set)
        encoded = encoder(model, hidden_layer, test_set)
        decoded = decoder(encoded, model, hidden_layer)
        test_loss = np.mean((test_set - decoded['a'][-1])**2)
    avg += test_loss
    avg = avg/5
    if (avg < best_loss):
        best_loss = avg
        best_param = {'hidden_layers': hidden_layer,
                      'hidden_units': hidden_unit}
        print(best_param)

print('Best loss: {0}'.format(best_loss))
print('Best param: {0}'.format(best_param))
