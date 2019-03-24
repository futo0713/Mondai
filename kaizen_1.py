import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#data set
num_of_sam = 40
radius = 4
std_dv = 0.6

X_center = np.random.randn(num_of_sam,2)*std_dv

s = np.random.uniform(0,2*np.pi,num_of_sam)
noise = np.random.uniform(0.9, 1.1, num_of_sam)
x1 = np.sin(s)*radius*noise
x2 = np.cos(s)*radius*noise
X_circle = np.c_[x1,x2]

X = np.vstack((X_center,X_circle))

t_group1 = np.tile([0,1],(num_of_sam,1))
t_group2 = np.tile([1,0],(num_of_sam,1))

T = np.vstack((t_group1, t_group2))

#function1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def loss(y, t):
    return -np.sum(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

#function2
def forward(X,T,params):
    W1,W2,B1,B2 = params
    H = sigmoid(np.dot(X,W1)+B1)
    Y = softmax(np.dot(H,W2)+B2)
    E = loss(Y, T)

    dW2 = np.dot(H.T,Y-T)
    dB2 = np.sum(Y-T, axis=0, keepdims=True)
    dW1 = np.dot(X.T,H*(1-H)*np.dot(Y-T,W2.T))
    dB1 = np.sum(H*(1-H)*np.dot(Y-T,W2.T), axis=0, keepdims=True)
    dE = [dW1, dW2, dB1, dB2]
    return [H, Y, E, dE]

def velocity(learning_rate,momentum_term,Vs,dE):
    return [momentum_term*i-learning_rate*j for i, j in zip(Vs ,dE)]

def momentum(Vs,params):
    return [i+j for i, j in zip(params, Vs)]

#initialize
W1 = np.random.randn(2,3)
W2 = np.random.randn(3,2)
B1 = np.random.randn(1,3)
B2 = np.random.randn(1,2)
params_momentum = [W1,W2,B1,B2]

learning_rate = 0.008
momentum_term = 0.9
Vs = [np.zeros_like(i) for i in [W1,W2,B1,B2]]

save_momentum =[]

#iteration
num_of_itr = 600
for i in range(num_of_itr):
    output_momentum = forward(X,T,params_momentum)
    Vs = velocity(learning_rate,momentum_term,Vs,output_momentum[3])
    params_momentum = momentum(Vs,params_momentum)

    #save loss
    save_momentum = np.append(save_momentum, output_momentum[2])

#3D plot
group1_hidden,group2_hidden = np.split(output_momentum[0],2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('H1')
ax.set_ylabel('H2')
ax.set_zlabel('H3')
ax.scatter(group1_hidden[:,0], group1_hidden[:,1], group1_hidden[:,2])
ax.scatter(group2_hidden[:,0], group2_hidden[:,1], group2_hidden[:,2])
plt.show()
