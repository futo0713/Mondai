import numpy as np
import matplotlib.pyplot as plt

#dataset
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

#function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def loss(y, t):
    return -np.sum(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

#initial parameter setting
W1 = np.random.randn(2,3)
B1 = np.random.randn(1,3)
W2 = np.random.randn(3,2)
B2 = np.random.randn(1,2)

velocity_W1 = np.zeros_like(W1)
velocity_B1 = np.zeros_like(B1)
velocity_W2 = np.zeros_like(W2)
velocity_B2 = np.zeros_like(B2)

learning_rate = 0.008
momentum_term = 0.9
E_save = []

#iteration
num_of_itr = 600
for i in range(num_of_itr):
    #forward propagation
    H = sigmoid(np.dot(X,W1)+B1)
    Y = softmax(np.dot(H,W2)+B2)
    E = loss(Y, T)
    E_save = np.append(E_save, E)
    #back propagation
    dW2 = np.dot(H.T,Y-T)
    dB2 = np.sum(Y-T, axis=0, keepdims=True)

    dW1 = np.dot(X.T,H*(1-H)*np.dot(Y-T,W2.T))
    dB1 = np.sum(H*(1-H)*np.dot(Y-T,W2.T), axis=0, keepdims=True)

    #update
    velocity_W1 = momentum_term*velocity_W1-learning_rate*dW1
    W1 = W1+velocity_W1

    velocity_B1 = momentum_term*velocity_B1-learning_rate*dB1
    B1 = B1+velocity_B1

    velocity_W2 = momentum_term*velocity_W2-learning_rate*dW2
    W2 = W2+velocity_W2

    velocity_B2 = momentum_term*velocity_B2-learning_rate*dB2
    B2 = B2+velocity_B2

#plot
grid_range = 10
resolution = 50
x1_grid = x2_grid = np.linspace(-grid_range, grid_range, resolution)

xx, yy = np.meshgrid(x1_grid, x2_grid)
X_grid = np.c_[xx.ravel(), yy.ravel()]

H_grid = sigmoid(np.dot(X_grid, W1)+B1)
Y_grid = softmax(np.dot(H_grid, W2)+B2)
Y_predict = np.around(Y_grid)

out_connect = np.hstack((X_grid,Y_predict))
blue_group = out_connect[out_connect[:,2]==1]
red_group = out_connect[out_connect[:,3]==1]

#output
plt.figure()

plt.title("Deep Learning")
plt.xlabel("input(x1)")
plt.ylabel("input(x2)")

plt.grid(True)
plt.xlim(-grid_range,grid_range)
plt.ylim(-grid_range,grid_range)
plt.plot(blue_group[:,0],blue_group[:,1],'o',alpha=0.3,color='blue')
plt.plot(red_group[:,0],red_group[:,1],'o',alpha=0.3,color='red')

plt.plot(X_center[:,0],X_center[:,1], 'o',color='red')
plt.plot(X_circle[:,0],X_circle[:,1], 'o',color='blue')
plt.show()

#loss
plt.figure()

plt.title("Loss Function")
plt.xlabel("iteration number")
plt.ylabel("loss value")

plt.grid(True)
plt.xlim(0,num_of_itr)
plt.ylim(0,50)

plt.plot(E_save)
plt.show()
