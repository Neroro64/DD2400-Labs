import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GaussianNoise
from keras import regularizers

seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

# Saves the 1510 first values of the time series
xs = np.zeros((1510))

def x(t, beta=0.2, gamma=0.1, n=10, tau=25, max_t = 1510):
    
    # Handle negative t's
    if t < 0:
        return 0
    # Handle t = 0
    if t == 0:
        xs[0] = 1.5
        return xs[0] 
    
    # Compute x(t-1)
    if t < max_t and xs[t-1] != 0:
        xtm1 = xs[t-1]  
    else:
        xtm1 = x(t-1)
        
    # Compute x(t-tau)    
    if t < tau:
        xtm25 = 0
    else:        
        if xs[t-tau] != 0:
            xtm25 = xs[t-tau]        
        else: 
            xtm25 = x(t-tau)
        
    # Compute the result
    euler = xtm1 + (beta*xtm25/(1+xtm25**n)) - gamma*xtm1
    if t < max_t:
        xs[t] = euler
        
    return euler



def split_inputs(input_patterns, targets, proportion):
        
    S = np.shape(input_patterns)[1]
    
    test_patterns = np.zeros((5, proportion))
    test_targets = np.zeros((1, proportion))
    data_patterns = np.zeros((5, S-proportion))
    data_targets = np.zeros((1, S-proportion))
    
    for i in range(proportion):
        test_patterns[0][i] = input_patterns[0][S-proportion+i]
        test_patterns[1][i] = input_patterns[1][S-proportion+i]
        test_patterns[2][i] = input_patterns[2][S-proportion+i]
        test_patterns[3][i] = input_patterns[3][S-proportion+i]
        test_patterns[4][i] = input_patterns[4][S-proportion+i]
        test_targets[0][i] = targets[S-proportion+i]

    for j in range(S-proportion):
        data_patterns[0][j] = input_patterns[0][j]
        data_patterns[1][j] = input_patterns[1][j]
        data_patterns[2][j] = input_patterns[2][j]
        data_patterns[3][j] = input_patterns[3][j]
        data_patterns[4][j] = input_patterns[4][j]
        data_targets[0][j] = targets[j]

    
    return data_patterns, data_targets, test_patterns, test_targets



    
    
def train(patterns, targets, nb_hidden_layers, nb_hidden_nodes, noise, regul_strength=0.00001, do_plot=True):
    np.random.seed(2020)
    model = network(nb_hidden_layers, nb_hidden_nodes, noise, regul_strength=regul_strength)
    
    # Train the network and save results
    history = model.fit(np.transpose(patterns), np.transpose(targets), verbose=0, epochs=300, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=False)])

    l1 = history.history['loss']
    l2 = history.history['val_loss']

    l1.pop(0)
    l2.pop(0)
    
    if do_plot:
        plt.plot(l1)
        plt.plot(l2)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
        
    print("Lowest training MSE: " + str(min(l1)))
    print("Lowest validation MSE: " + str(min(l2)))
        
    return model, history



def network(nb_hidden_layers, nb_hidden_nodes, noise, regul_strength=0.00001):
    np.random.seed(2020)
    model = Sequential()
    
    # Without noise
    if noise == -1:
        # 2 layer net
        if nb_hidden_layers == 1:
            model.add(Dense(nb_hidden_nodes[0],
                            input_shape=(5,),
                            use_bias=True,
                            activation='sigmoid',
                            kernel_regularizer=regularizers.l1(regul_strength)))
        
        # 3 layer net
        elif nb_hidden_layers == 2:
             model.add(Dense(nb_hidden_nodes[0],
                            input_shape=(5,),
                            use_bias=True,
                            activation='sigmoid',
                            kernel_regularizer=regularizers.l1(regul_strength)))
             
             model.add(Dense(nb_hidden_nodes[1],
                            use_bias=True,
                            activation='sigmoid',
                            kernel_regularizer=regularizers.l1(regul_strength)))
             
    else:
        # 2 layer net
        if nb_hidden_layers == 1:
            model.add(GaussianNoise(noise))
            model.add(Dense(nb_hidden_nodes[0],
                            input_shape=(5,),
                            use_bias=True,
                            activation='sigmoid',
                            kernel_regularizer=regularizers.l1(regul_strength)))
            
        
        # 3 layer net
        elif nb_hidden_layers == 2:
             model.add(GaussianNoise(noise))
             model.add(Dense(nb_hidden_nodes[0],
                            input_shape=(5,),
                            use_bias=True,
                            activation='sigmoid',
                            kernel_regularizer=regularizers.l1(regul_strength)))
             
             model.add(Dense(nb_hidden_nodes[1],
                            use_bias=True,
                            activation='sigmoid',
                            kernel_regularizer=regularizers.l1(regul_strength)))

    # Add the output layer
    model.add(Dense(1, use_bias=True, kernel_regularizer=regularizers.l1(regul_strength)))
    
    # Compile the model
    model.compile(optimizer=SGD(lr=0.05, momentum=0.9, nesterov=False), loss='mean_squared_error', metrics=['accuracy'])
    
    return model


def run(nb_hidden_layers, nb_hidden_nodes, data_patterns, data_targets, noise):
    np.random.seed(2020)
    model, _ = train(data_patterns, data_targets, nb_hidden_layers, nb_hidden_nodes, noise)

    return model



def plot_hist(data_patterns, data_targets, nb_hidden_layers, nb_hidden_nodes, reg_strengths):
    
    for i in range(len(reg_strengths)):
        model, _ = train(data_patterns, data_targets, nb_hidden_layers, nb_hidden_nodes, -1, regul_strength=reg_strengths[i])        
        weights = model.get_weights()
        # Get the weights (without biases)
        w = weights[0] 
        v = weights[2]
        plt.title('Weights histogram for regularisation strength = ' + str(reg_strengths[i]))
        plt.xlim(-2, 2)
        plt.hist(np.concatenate((v.flatten(), w.flatten())))
        #plt.hist(np.concatenate((v.flatten(), w.flatten())), bins=np.arange(-2.0, 2.2, 0.2) if i == 0 else 'auto')
        plt.show()
        
# Compute test MSE        
def test_MSE(A,B):
   mse = ((A - B)**2).mean(axis=None)
   return mse 


#############################
    
######### Data ##########
n_data = 1200
t = np.arange(301, 1501, 1)
input_patterns = np.zeros((6, n_data))

for i in range(n_data):
    input_patterns[0][i] += x(t[i]-20)
    input_patterns[1][i] += x(t[i]-15)
    input_patterns[2][i] += x(t[i]-10)
    input_patterns[3][i] += x(t[i]-5)
    input_patterns[4][i] += x(t[i])
    # Targets
    input_patterns[5][i] = x(t[i]+5) 
    
input_patterns, targets = np.split(input_patterns, [5])

# Split data into (train, validation) and testing.
data_patterns, data_targets, test_patterns, test_targets = split_inputs(input_patterns, targets.flatten(), 200)


########## Plot test predictions vs actual time series ############
#model = run(1, [5], data_patterns, data_targets)
model = run(2, [6, 6], data_patterns, data_targets, -1)

predictions = model.predict(np.transpose(test_patterns))
predictions = predictions.reshape(len(predictions))
time = np.arange(1300, 1500, 1)
plt.plot(time, test_targets.flatten())
plt.plot(time, predictions)
plt.legend(['Time series', 'Approximation'])
plt.show()

print("Test MSE for 3-layer is: " + str(test_MSE(test_targets.flatten(),predictions)))

########## Histograms ##########
#reg_strengths = [0.001, 0.00001]
#nb_hidden_layers = 1
#nb_hidden_nodes = [6]
#plot_hist(data_patterns, data_targets, nb_hidden_layers, nb_hidden_nodes, reg_strengths)


######## Noise ########
np.random.seed(2020)

# Add noise
noise = [0.03, 0.09, 0.18]


########## Plot test predictions vs actual time series ############
#model = run(1, [5], data_patterns, data_targets)
model = run(2, [6, 6], data_patterns, data_targets, noise[1])

predictions = model.predict(np.transpose(test_patterns))
predictions = predictions.reshape(len(predictions))
time = np.arange(1300, 1500, 1)
plt.plot(time, test_targets.flatten())
plt.plot(time, predictions)
plt.legend(['Time series', 'Approximation'])
plt.title('3-layer with noise')
plt.show()


print("Test MSE for 3-layer with noise is: " + str(test_MSE(test_targets.flatten(),predictions)))
