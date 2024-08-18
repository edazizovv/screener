#

#
import numpy
from matplotlib import pyplot

import torch
from torch import nn

#


#
class Test(nn.Module):
    def __init__(self, n_hidden, epochs, optimiser, optimiser_kwargs, loss_function):
        super().__init__()
        self.n_hidden = n_hidden
        self._epochs = epochs

        self._gru = torch.nn.GRU
        self._ln = torch.nn.Linear
        self._relu = torch.nn.LeakyReLU

        self.gru = None
        self.ln = None
        self.relu = None

        self._optimiser = optimiser
        self._optimiser_kwargs = optimiser_kwargs

        self.optimiser = None
        self.aggregated_losses = []
        self.validation_losses = []

        self.loss_function = loss_function()

    def fit(self, X_train, X_val, Y_train, Y_val):
        self.gru = self._gru(X_train.shape[2], self.n_hidden, 1, batch_first=True)
        self.ln = self._ln(self.n_hidden, 1)
        self.relu = self._relu()

        self.optimiser = self._optimiser(self.parameters(), **self._optimiser_kwargs)

        self.aggregated_losses = []
        self.validation_losses = []

        for i in range(self._epochs):
            i += 1
            for phase in ['train', 'validate']:

                if phase == 'train':
                    y_pred = self(X_train)
                    single_loss = self.loss_function(y_pred, Y_train)
                else:
                    y_pred = self(X_val)
                    single_loss = self.loss_function(y_pred, Y_val)

                self.optimiser.zero_grad()

                if phase == 'train':
                    train_lost = single_loss.item()
                    self.aggregated_losses.append(single_loss)
                    single_loss.backward()
                    self.optimiser.step()
                else:
                    validation_lost = single_loss.item()
                    self.validation_losses.append(single_loss)

            if i % 25 == 1:
                print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost,
                                                                                             validation_lost))
        print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost, validation_lost))

    def forward(self, x):

        out, h = self.gru(x)
        out = self.ln(out)
        out = self.relu(out)
        out = out[:, -1, :]

        return out

    def predict(self, X):

        output = self(X)
        result = output.detach().numpy()

        return result




class NN(nn.Module):

    def __init__(self,
                 layers, layers_dimensions, layers_kwargs,
                 batchnorms, activators,
                 interdrops,
                 optimiser, optimiser_kwargs,
                 epochs, device=None
                 ):

        self.layers = None
        self.input_size = None
        self.optimiser = None
        self.batch_size = None
        self.train_loss = []
        self.validation_loss = []

        self._layers = layers
        self._layers_dimensions = layers_dimensions
        self._layers_kwargs = layers_kwargs
        self._batchnorms = batchnorms
        self._activators = activators
        self._interdrops = interdrops
        self._optimiser = optimiser
        self._optimiser_kwargs = optimiser_kwargs
        self._epochs = epochs
        self._aggregated_losses = None
        self._validation_losses = None

        self.device = device

        super().__init__()

    def initialise(self):

        all_layers = []

        input_size = self.input_size
        for j in range(len(self._layers_dimensions)):
            all_layers.append(self._layers[j](input_size, self._layers_dimensions[j], **self._layers_kwargs[j]))
            if not (not self._batchnorms[j]):
                all_layers.append(self._batchnorms[j](self._layers_dimensions[j]))
            if self._activators[j] is not None:
                if self._activators[j].__name__ == 'Softmax':
                    all_layers.append(self._activators[j](dim=1))
                else:
                    all_layers.append(self._activators[j]())
            if self._interdrops[j] is not None:
                all_layers.append(nn.Dropout(self._interdrops[j]))
            input_size = self._layers_dimensions[j]

        self.layers = nn.ModuleList(all_layers)
        if self.device is not None:
            self.layers.to(device=self.device)

        #

    def _requires_hidden_cell(self, j):
        if self.layers is None:
            return self._layers[j].__name__ in ['RNN', 'GRU', 'LSTM']
        else:
            return self.layers[j]._get_name() in ['RNN', 'GRU', 'LSTM']

    def _depack_hidden(self, y, j):
        if self.layers[j]._get_name() == 'LSTM':
            y, n = self.layers[j](y, self.hidden_cell(j))
            hn, cn = n
        elif self.layers[j]._get_name() in ['RNN', 'GRU']:
            y, hn = self.layers[j](y)
        else:
            raise Exception("Whoops...")
        return y

    def _is_technical(self, j):
        return self.layers[j]._get_name() in ['Dropout', 'BatchNorm1d']

    def hidden_cell(self, j):
        if self.layers[j]._get_name() == 'LSTM':
            if self.device:
                return torch.zeros(1, self.batch_size, self.layers[j].hidden_size, device=self.device).requires_grad_(), \
                       torch.zeros(1, self.batch_size, self.layers[j].hidden_size, device=self.device).requires_grad_()
            else:
                return torch.zeros(1, self.batch_size, self.layers[j].hidden_size).requires_grad_(), \
                       torch.zeros(1, self.batch_size, self.layers[j].hidden_size).requires_grad_()
        elif self.layers[j]._get_name() in ['RNN', 'GRU']:
            if self.device:
                return torch.zeros(1, self.batch_size, self.layers[j].hidden_size, device=self.device).requires_grad_()
            else:
                return torch.zeros(1, self.batch_size, self.layers[j].hidden_size).requires_grad_()

        else:
            raise Exception("Whoops...")

    def forward(self, x):

        #

        self.batch_size = x.shape[0]

        #

        y = x

        for j in range(len(self.layers)):
            if not self._is_technical(j):
                if not self._requires_hidden_cell(j):
                    if y.dim() == 2:
                        y = self.layers[j](y)
                    elif y.dim() == 3:
                        y = self.layers[j](y[:, -1, :])
                    else:
                        raise Exception("Whoops...")
                else:
                    # y, hidden_cell = self.layers[j](y, self.hidden_cell(j))
                    y = self._depack_hidden(y, j)
            else:
                y = self.layers[j](y)

        # this is bad !!

        # sm = nn.Softmax(dim=1)
        # y = sm(y)

        #

        return y

    def fit(self, X_train, y_train, X_val, y_val, loss_function):

        if self._requires_hidden_cell(0):
            self.input_size = X_train.shape[2]
        else:
            self.input_size = X_train.shape[1]

        self.initialise()
        self.optimiser = self._optimiser(self.parameters(), **self._optimiser_kwargs)

        self.train_loss = []
        self.validation_loss = []

        for i in range(self._epochs):
            i += 1
            for phase in ['train', 'validate']:

                if phase == 'train':
                    y_pred = self(X_train)
                    single_loss = loss_function(y_pred, y_train)
                else:
                    y_pred = self(X_val)
                    single_loss = loss_function(y_pred, y_val)

                self.optimiser.zero_grad()

                if phase == 'train':
                    train_lost = single_loss.item()
                    self.train_loss.append(train_lost)
                    single_loss.backward()
                    self.optimiser.step()
                else:
                    validation_lost = single_loss.item()
                    self.validation_loss.append(validation_lost)

            if i % 25 == 1:
                print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost,
                                                                                             validation_lost))
        print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost, validation_lost))

    def plot_fit(self):

        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.train_loss, label='Train')
        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.validation_loss, label='Validation')
        pyplot.legend(loc="upper left")
        pyplot.show()

    def predict(self, X_test):

        output = self(X_test)
        if self.device is not None:
            if self.device.type == 'cuda':
                output = output.cpu()
        result = output.detach().numpy()

        return result


class WrappedNN:

    def __init__(self,
                 layers, layers_dimensions, layers_kwargs,
                 batchnorms, activators,
                 optimiser, optimiser_kwargs, loss_function,
                 interdrops,
                 epochs=500,
                 device=None):
        self.optimiser = optimiser
        self.optimiser_kwargs = optimiser_kwargs
        self.loss_function = loss_function()

        self.model = NN(
            layers=layers, layers_dimensions=layers_dimensions, layers_kwargs=layers_kwargs,
            batchnorms=batchnorms, activators=activators,
            interdrops=interdrops,
            optimiser=optimiser, optimiser_kwargs=optimiser_kwargs,
            epochs=epochs, device=device)

    def fit(self, X_train, Y_train, X_val, Y_val):
        self.model.fit(X_train=X_train, y_train=Y_train, X_val=X_val, y_val=Y_val, loss_function=self.loss_function)

    def plot_fit(self):
        self.model.plot_fit()

    def predict(self, X):
        return self.model.predict(X)
