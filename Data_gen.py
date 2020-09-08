import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as ex
import plotly.graph_objects as go

np.random.seed(2020)

n = 100

meanA = [1.7, 1.7]
meanB = [-1.7, -1.7]
sigmaA = 0.5
sigmaB = 0.5

nl_meanA = [1.7, 1.7]
nl_meanB = [1, 1]
nl_sigmaA = 0.4
nl_sigmaB = 0.4


g_mA = [1.0, 0.3]
g_sigmaA = 0.2;
g_mB = [0.0, -0.1]
g_sigmaB = 0.3;



# As defined in the lab    
def linear(n, m1,m2, sigma1, sigma2):
    classA = np.random.normal(m1, sigma1, size=[n,2])
    classB = np.random.normal(m2, sigma2, size=[n,2])
    return classA, classB


# As defined in the lab 
def non_linear(n, m1,m2, sigma1, sigma2):
    
    classA = np.zeros((n,2))
    
    temp1 = (np.random.normal(-m1[0], sigma1, size=[int(round(0.5*n)),1]))
    temp2 = (np.random.normal(m1[0], sigma1, size=[int(round(0.5*n)),1]))
        
    temp_flat = np.squeeze(np.concatenate((temp1, temp2), axis=0), axis=1)
    classA[:,0] = temp_flat

    classA[:,1] = np.squeeze(np.random.normal(m1[1], sigma1, size=[n,1]), axis=1)   
    classB = np.random.normal(m2, sigma2, size=[n,2])
    return classA, classB


def plot_data(dataA, dataB): 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataA[:,0], y=dataA[:,1], mode="markers"))
    fig.add_trace(go.Scatter(x=dataB[:,0], y=dataB[:,1], mode="markers"))
    fig.show()  
 
    
# Linear data
dataA, dataB = linear(n, meanA, meanB, sigmaA, sigmaB)
plot_data(dataA, dataB)

# Non-linear data (overlapping distributions)
nl_dataA, nl_dataB = linear(n, nl_meanA, nl_meanB, nl_sigmaA, nl_sigmaB)
plot_data(nl_dataA, nl_dataB)

# Generated non-linear data
g_dataA, g_dataB = non_linear(n, g_mA, g_mB, g_sigmaA, g_sigmaB)
plot_data(g_dataA, g_dataB)