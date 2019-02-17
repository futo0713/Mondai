import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#dataset
num_of_sam = 40
std_dv = 1.8

group1 = np.array([2.5,2.5])+np.random.randn(num_of_sam, 2)*std_dv
group2 = np.array([7.5,7.5])+np.random.randn(num_of_sam, 2)*std_dv
X = np.vstack((group1, group2))

t_group1 = np.zeros((num_of_sam, 1))
t_group2 = np.ones((num_of_sam, 1))
t = np.vstack((t_group1, t_group2))

#function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(y, t):
    return -np.sum(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

#initial setting
W = np.random.randn(2,1)
b = np.random.randn(1,1)

learning_rate = 0.003
E_save = []

#iteration
num_of_itr = 500
for i in range(num_of_itr):
    #forward propagation
    y = sigmoid(np.dot(X, W)+b)
    E = loss(y, t)
    E_save = np.append(E_save, E)
    #back propagation
    dW = np.sum(X*(y-t),axis=0)
    db = np.sum(y-t)

    #update
    W = W - learning_rate*np.reshape(dW,(2,1))
    b = b - learning_rate*db

#plot
grid_range = 10
resolution = 60
x1_grid = x2_grid = np.linspace(-grid_range, grid_range, resolution)

xx, yy = np.meshgrid(x1_grid, x2_grid)
X_grid = np.c_[xx.ravel(), yy.ravel()]

Y_grid = sigmoid(np.dot(X_grid, W)+b)
Y_predict = np.around(Y_grid)

#plot_output
plt.figure()
plt.grid(True)
plt.title("Deep Learning")
plt.xlabel("input(x1)")
plt.ylabel("input(x2)")
plt.xlim(-grid_range, grid_range)
plt.ylim(-grid_range, grid_range)
plt.scatter(X_grid[:,0], X_grid[:,1], vmin=0, vmax=1, c=Y_predict[:,0], cmap=cm.bwr, marker='o', s=50,alpha=0.2)
plt.scatter(X[:,0], X[:,1], vmin=0, vmax=1, c=t[:,0], cmap=cm.bwr, marker='o', s=50)
plt.show()

#plot_loss
plt.figure()
plt.grid(True)
plt.title("LOSS FUNCTION")
plt.xlabel("Iteration number")
plt.ylabel("loss value")
plt.plot(E_save)
plt.show()



