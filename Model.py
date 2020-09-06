"""
#################################################################
Assignment 2
#################################################################
"""
__author__ = "Nuo Chen"

import numpy as np
import functions as f
from matplotlib import pyplot as plt
import json
import sys
import pickle


class Model:
    def __init__(self, nodes):
        """
        Arg:
            nodes - [in, n1, n2, ..., nj, out] an array of numbers of nodes in each layer. The first layer is an input layer, and the last is the output layer. 
        Returns:
            a fully-connected network with default weights and biases sampled from N(0, 1)
        """
        self.L = len(nodes)
        self.Nodes = nodes
        self.init_param()

    def init_param(self):
        """
        Initialize the weights and biases with samples from normal distributions
        W = [(n1xn2), (n2xn3), ..., (nj,nj+1)], where nj = number of nodes
        B = [(n1x1), (n2x1), ..., (njx1)]
        """
        self.W = [np.random.normal(0, 1, (i, j)) for i, j in zip(self.Nodes[:-1], self.Nodes[1:])]
        self.B = [np.random.normal(0, 1, (i,1)) for i in self.Nodes[1:]]

    def init_training_param(self, n, batch_size, epochs, lmbda, eta_min, eta_max, eta_size=0, cycles=0):
        """
        Initialize the hyper-parameters
        """
        self.N = n
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs if eta_size == 0 else int(cycles * 2 * eta_size)
        self.ETA_MIN = eta_min
        self.ETA_MAX = eta_max
        self.ETA_SIZE = eta_size
        self.BATCHES = int(n / batch_size)
        self.LAMBDA = lmbda
        self.ETA = eta_min

    def normalize(self, x):
        """
        Normalize the input data
        """
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        x = (x - mean) / std
        return x
        
    def softmax(self, x):
        return np.exp(x-np.max(x, axis=0)) / np.sum(np.exp(x-np.max(x, axis=0)), axis=0)
    
    def cross_entropy(self, p, y):
        """
        Return the cross entroy cost of the prediction
        Note: a pseudo probability is assigned to where the predictions is zero, this is to avoid the invalid value error.
        """
        p[p==0] = 1e-7
        cost = 1/y.shape[1] * -np.sum(y*np.log(p))
        w_sum = [w**2 for w in self.W]
        s = 0
        for w in w_sum: 
            s += np.sum(w)
        cost += self.LAMBDA * s

        return cost

    def update_eta(self, t, l):
        """
        Update the learning rate for each update step
        """
        self.ETA = (t - 2*l*self.ETA_SIZE) / self.ETA_SIZE
        self.ETA = self.ETA * (self.ETA_MAX - self.ETA_MIN) + self.ETA_MIN

    def onehot(self, y):
        """
        converts y to one hot encoding
        """
        onehot = np.zeros((np.max(y)+1, len(y)))
        onehot[y, np.arange(len(y))] = 1
        return onehot
    
    def feedforward(self, activations):
        """
        s1 = w1 @ x + b1
        h1 = max(0, s1)
        ...
        sn = wn @ hn + bn
        return Softmax(sn)
        """

        a = activations[0]
        for i in range(self.L-1):
            s = self.W[i].T @ a + self.B[i]
            a = np.maximum(0, s)
            activations.append(a)

        p = self.softmax(a)
        return p;

    def backPropagation(self, y, p, activations):
        """
        Back propagate the network and calculate the gradients
        """
        dw = [np.zeros(w.shape) for w in self.W]
        db = [np.zeros(b.shape) for b in self.B]

        g = -(y - p)
        for i in range(len(self.W)-1, -1, -1):
            dw[i] = g @ activations[i].T * 1/self.BATCH_SIZE + 2 * self.LAMBDA * self.W[i].T
            db[i] =  (np.sum(g, axis=1) * 1/self.BATCH_SIZE).reshape(self.B[i].shape)
            g = self.W[i] @ g
            g[np.where(activations[i]<=0)] = 0
        
        return (dw, db)
    
    def accuracy(self, p, y):
        """
        Compute the accuracy of the predictions
        """
        predictions = np.argmax(p, axis=0)
        y = np.argmax(y, axis=0)
        acc = predictions.T[predictions == y].shape[0] / p.shape[1]
        return acc
    
    def update_batch(self, x, y):
        """
        For each batch: 
            Pass the input into the network and compute the predictions.
            Back propagate through the network to compute the gradients using the stored activations
            Update the weights and biases using the gradients
        """
        activations = [x]
        p = self.feedforward(activations)
        dw, db = self.backPropagation(y, p, activations)

        for i in range(self.L-1):
            self.W[i] = self.W[i] - self.ETA * dw[i].T
            self.B[i] = self.B[i] - self.ETA * db[i]
        

    def SGD(self, training_data, test_data, log=False):
        """
        Stochastic gradient descend method
        Trains the network a given number of epochs or cycles
        Return:
            Training cost and validation cost
            Training accuracy and validation accuracy
        """
        x_t = training_data[0]
        x_v = test_data[0]
        y_t = training_data[1]
        y_v = test_data[1]

        training_cost = []
        validation_cost = []
        training_accuracy = []
        validation_accuracy = []

        t = 0;
        k = (self.ETA_SIZE * 2) / 10
        while(t < self.EPOCHS):
            # Shuffles the order of samples 
            idx = np.random.permutation(self.N)

            for j in range(1, self.BATCHES):
                t += 1
                l = np.floor(t / (2 * self.ETA_SIZE))
                start = (j-1) * self.BATCH_SIZE
                end = j * self.BATCH_SIZE
                indices = idx[start:end]
                x_batch = x_t[:, indices]            
                y_batch = y_t[:, indices]            
                self.update_batch(x_batch, y_batch)
                self.update_eta(t, l)

                # Check cost and accuracy 10 times per cycle 
                if (t % k == 0):
                    p_t = self.feedforward([x_t])
                    p_v = self.feedforward([x_v])
                    training_cost.append(self.cross_entropy(p_t, training_data[1]))
                    validation_cost.append(self.cross_entropy(p_v, test_data[1]))
                    training_accuracy.append(self.accuracy(p_t, training_data[1]))
                    validation_accuracy.append(self.accuracy(p_v, test_data[1]))

            if (log):
                print("Epoch #{}--------------------------------------".format(i))
                print("Training Cost: {:.6f}".format(training_cost[-1]))
                print("Validation Cost: {:.6f}".format(validation_cost[-1]))
                print("Training Accuracy = {:.3f}".format(training_accuracy[-1]))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy[-1]))
                print("-"*50)

        return (training_cost, validation_cost, training_accuracy, validation_accuracy)

    def save(self, filename):
        """
        Save the model to the file 'filename`.
        """
        data = {"Nodes": self.Nodes,
                "W": [w.tolist() for w in self.W],
                "B": [b.tolist() for b in self.B]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def load(self, filename):
        """
        Load the model
        """
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        self.Nodes = data["Nodes"]
        self.W = [np.array(w) for w in data["W"]]
        self.B = [np.array(b) for b in data["B"]]

    def compute_gradients_num(self, x, y, h=1e-7):
        """
        Converted from MatLab code
        """
        dW = [np.zeros(w.shape) for w in self.W]
        dB = [np.zeros(b.shape) for b in self.B]
        
        c = self.cross_entropy(self.feedforward([x]), y)
        for i in range(len(self.B)):
            b = self.B[i]    
            for j in range(len(b)):
                self.B[i][j] += h
                c2 = self.cross_entropy(self.feedforward([x]),y)
                dB[i][j] = (c2-c) / h

        for i in range(len(self.W)):
            w = self.W[i]    
            for j in range(len(w)):
                for k in range(len(w[j])):
                    self.W[i][j,k] += h
                    c2 = self.cross_entropy(self.feedforward([x]), y)
                    dW[i][j,k] = (c2-c) / h
        
        return dW, dB
    
    def validate_gradient(self, x, y):
        """
        Check if the analytically computed gradients match the numerically computed gradients
        """
        activations = [x]
        p = self.feedforward(activations)
        dw, db = self.backPropagation(y, p, activations)

        dw2, db2 = self.compute_gradients_num(x, y)
        diff_w = [np.abs(w2.T - w1) for w2, w1 in zip(dw2, dw)]
        diff_w = np.ravel(np.asarray([np.where(w < 1e-6, 0, 1) for w in diff_w]))
        print("diff W = " + str(np.sum(diff_w)/diff_w.size))
        
        diff_b = [np.abs(b2.T - b1) for b2, b1 in zip(db2, db)]
        diff_b = np.ravel(np.asarray([np.where(b < 1e-6, 0, 1) for b in diff_b]))
        print("diff B = " + str(np.sum(diff_b)/diff_b.size))

"""
############################################################################
"""

def load(filename):
    """ 
    Loads a dataset and returns inputs, outputs and the corresponding labels
    """
    with open(filename, 'rb') as f:
        dataDict = pickle.load(f, encoding='bytes')

    x = (dataDict[b"data"]).T
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)

    x = (x - mean) / std
    y = np.array(dataDict[b"labels"])
    y_onehot = (np.eye(10)[y]).T
    return [x, y_onehot, y]

"""
Setting the parameters
"""
ETA_MAX = 1e-1
ETA_MIN = 1e-5 
BATCH_SIZE = 100
EPOCH = 40
LMBDA = .01
CYCLES = 12

"""
Loading the training and validation datasets
"""
np.random.seed(0)
x1, y1, yl1 = load("data_batch_1")
x2, y2, yl2 = load("data_batch_2")
x3, y3, yl3 = load("data_batch_3")
x4, y4, yl4 = load("data_batch_4")
x5, y5, yl5 = load("data_batch_5")
x_v, y_v, yl = load("test_batch")

# """
# Limiting training size to 5000 samples, and validation size = 500 samples 
# # """
# x_t = x1[:, :5000]
# y_t = y1[:, :5000]
# x_v = x1[:, 9000:9500]
# y_v = y1[:, 9000:9500]

"""
Use all samples for training, except the last 5000 samples which are used for validation
"""
x_t = np.concatenate((x1,x2,x3,x4,x5,x_v[:, :9000]), axis=1)
y_t = np.concatenate((y1,y2,y3,y4,y5,y_v[:, :9000]), axis=1)
x_v = x_v[:, 9000:]
y_v = y_v[:, 9000:]

"""
Compute the step size
"""
ETA_SIZE = 2 * np.floor(x_t.shape[1]/BATCH_SIZE)
LAMBDA_MIN = -5
LAMBDA_MAX = -1
TEST = 10
"""
Draw a number of random lambda from the range (LAMBDA_MIN, LAMBDA_MAX) and store the accuracy of the network after training
"""
val_acc = []
lambdas = []
for i in range(1):
    # LMBDA = np.power(10, LAMBDA_MIN + (LAMBDA_MAX-LAMBDA_MIN)*np.random.random())
    # LMBDA = np.random.normal(0.004)
    LMBDA = 0.0398
    """
    Setting up the 2-layer network. Input layer (3072 nodes), hidden layer (50 nodes) and output layer (10 nodes) 
    """
    model = Model([3072, 50, 10]) 
    model.init_param()
    model.init_training_param(x_t.shape[1], BATCH_SIZE, EPOCH, LMBDA, ETA_MIN, ETA_MAX, ETA_SIZE, CYCLES)
    # model.validate_gradient(x_t, y_t)
    tCost, vCost, tAcc, vAcc = model.SGD([x_t, y_t], [x_v, y_v])
    
    """
    Since the cost can be quite large, it must be converted to log scale
    """
    tCost = np.log(tCost)
    vCost = np.log(vCost)
    
    """
    Stores the lambda and the best accuracy
    """
    lambdas.append(LMBDA)
    val_acc.append(np.max(vAcc))

    """
    Plot the results and save the figures.
    """
    xaxis = np.arange(len(tCost))* ((2*ETA_SIZE)/10)
    plt.figure(figsize=(10,10))
    plt.subplot(211)
    plt.xlabel('Update step', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.plot(xaxis, tCost, label="Training set error")
    plt.plot(xaxis, vCost, label="Validation set error")
    plt.legend()

    plt.subplot(212)
    plt.xlabel('Update step', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.plot(xaxis, tAcc, label="Training Accuracy")
    plt.plot(xaxis, vAcc, label="Validation Accuracy")
    plt.legend()
    plt.savefig("search_lmbda={}_etaSize={}_etaMin={}_etaMax={}_acc={}.png".format(LMBDA, ETA_SIZE,ETA_MIN, ETA_MAX, np.max(vAcc)))
    model.save("search_model_lmbda={}_etaSize={}_etaMin={}_etaMax={}_acc={}".format(LMBDA, ETA_SIZE,ETA_MIN, ETA_MAX, np.max(vAcc)))
    # print("Best training accuracy: {}%".format(np.max(tAcc)))
    # print("Best validation accuracy: {}%".format(np.max(vAcc)))
    # plt.show()


print("Best validation accuracy: {}".format(np.max(val_acc)))
print("Best lambda: {}".format(lambdas[np.argmax(val_acc)]))
