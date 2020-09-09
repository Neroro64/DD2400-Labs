import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

np.random.seed(2020)

n = 100
epoch = 20
eta = 0.0001


# As defined in the lab    
def linear(n, m1,m2, sigma1, sigma2):
    classA = np.random.normal(m1, sigma1, size=[n,2])
    classB = np.random.normal(m2, sigma2, size=[n,2])
    return classA, classB


def plot_data(dataA, dataB): 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataA[:,0], y=dataA[:,1], mode="markers"))
    fig.add_trace(go.Scatter(x=dataB[:,0], y=dataB[:,1], mode="markers"))
    fig.show() 

# Not done yet    
def plot_boundary(dataA, dataB, W): 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataA[:,0], y=dataA[:,1], mode="markers"))
    fig.add_trace(go.Scatter(x=dataB[:,0], y=dataB[:,1], mode="markers"))
    
    lin_x = np.linspace(-5, 5, 100)
    y = np.sum((lin_x.reshape(100,1)@W[:, 0].reshape((1,W.shape[0]))) / W[:, 1], axis=1) / W.shape[0]
    # y = np.clip(y, -5, 5)
    fig.add_trace(go.Scatter(x=lin_x, y=y, mode="markers"))
    fig.show() 
    

class one_layer_model():
    def __init__(self, N):
        self.N = N
    def init_param(self, eta):
        self.ETA = eta

    def batch(self, input_vec, targets, epochs):
        #W = np.zeros((targets.shape[1], input_vec.shape[0]))
        W = np.random.uniform(0,1,((targets.shape[1], input_vec.shape[0])))
        for e in range(epochs):
            # W*X
            h = W @ input_vec
            # Threshold
            f = np.where(h > 0, 1, -1)
            W -= self.ETA*(f-targets) @ input_vec.T
            
            #mse = MSE(W, features, targets)
            #print(mse)
            
            #acc = accuracy(W, features, targets)
            #print(acc)
        return W, 0
            
        
             
def single_perceptron_learning(x, t, n, epochs, eta):
    model = one_layer_model(n)
    model.init_param(eta)

    model, mae = model.batch(x, t, epochs)
    
    return model, mae

def accuracy(W, input_vec, target):
    # Forward pass
    h = W @ input_vec
    f = np.where(h > 0, 1, -1)
    
    total = target.shape[1]
    
    count = 0
    
    for i in range(total):
        if target[0][i] == f[0][i]:
            count += 1
           
    return (count/total)*100

def MSE(W, input_vec, target):
    # Forward pass
    h = W @ input_vec
    f = np.where(h > 0, 1, -1)
    
    square = 0
    total = target.shape[1]
    
    for i in range(total):
        square += (target[0][i] - f[0][i])**2
           
    return square/total

(A, B) = linear(n, [1,0.5], [-1,0], 0.3, 0.3)


features = (np.concatenate([A,B],axis=0))
features = (np.concatenate([features,np.ones((features.shape[0],1))],axis=1)).T
targets = (np.concatenate([-np.ones((A.shape[0],1)), np.ones((B.shape[0],1))], axis=0)).T

W, mae = single_perceptron_learning(features, targets, 0, epoch, eta)

acc = accuracy(W, features, targets)
print(acc)


#plot_data(A,B)

#plot_boundary(A,B,W)












