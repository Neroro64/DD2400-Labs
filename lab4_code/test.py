import numpy as np
import plotly.express as ex
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
from util import *

def plotErr(mae):
    fig = ex.line(x=np.arange(len(mae)), y=mae)
    fig.show()

image_size = [28,28]
train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

''' restricted boltzmann machine '''

print ("\nStarting a Restricted Boltzmann Machine..")

rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                    ndim_hidden=200,
                                    is_bottom=True,
                                    image_size=image_size,
                                    is_top=False,
                                    n_labels=10,
                                    batch_size=20
)

mae = rbm.cd1(visible_trainset=train_imgs, n_iterations=10, logErr=True)
plotErr(mae)