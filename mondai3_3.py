import numpy as np
import matplotlib.pyplot as plt

#dataset
num_of_sam = 40
std_dv = 1.8

group1 = np.array([2.5,2.5])+np.random.randn(num_of_sam, 2)*std_dv
group2 = np.array([7.5,7.5])+np.random.randn(num_of_sam, 2)*std_dv
X = np.vstack((group1, group2))

t_group1 = np.tile([0,1],(num_of_sam,1))
t_group2 = np.tile([1,0],(num_of_sam,1))
T = np.vstack((t_group1, t_group2))


#function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def loss(y, t):
    return -np.sum(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

#initial setting
W = np.random.randn(2,2)
B = np.random.randn(1,2)

learning_rate = 0.003
E_save = []

#iteration
num_of_itr = 400
for i in range(num_of_itr):
    #forward propagation
    Y = softmax(np.dot(X, W)+B)
    E = loss(Y, T)
    E_save = np.append(E_save, E)
    #back propagation
    dW = X.T.dot(Y-T)
    dB = np.sum(Y-T, axis=0, keepdims=True)
    #update
    W = W - learning_rate*dW
    B = B - learning_rate*dB

#plot_grid
grid_range = 10
resolution = 60
x1_grid = x2_grid = np.linspace(-grid_range, grid_range, resolution)

xx, yy = np.meshgrid(x1_grid, x2_grid)
X_grid = np.c_[xx.ravel(), yy.ravel()]

Y_grid = softmax(np.dot(X_grid, W)+B)
Y_predict = np.around(Y_grid)

out_connect = np.hstack((X_grid,Y_predict))
red_group = out_connect[out_connect[:,2]==1]
blue_group = out_connect[out_connect[:,3]==1]

#plot_output
plt.figure()
plt.grid(True)
plt.xlim(-grid_range,grid_range)
plt.ylim(-grid_range,grid_range)

plt.scatter(red_group[:,0],red_group[:,1],marker='o',s=50,alpha=0.3,color='red')
plt.scatter(blue_group[:,0],blue_group[:,1],marker='o',s=50,alpha=0.3,color='blue')

plt.scatter(group1[:,0],group1[:,1],marker='o',s=50,color='blue')
plt.scatter(group2[:,0],group2[:,1],marker='o',s=50,color='red')
plt.show()

#plot_loss
plt.figure()

plt.title("Loss Function")
plt.xlabel("iteration number")
plt.ylabel("loss value")

plt.grid(True)
plt.xlim(0,num_of_itr)
plt.ylim(0,50)

plt.plot(E_save)
plt.show()
