import numpy as np
import matplotlib.pyplot as plt

#dataset
num_of_sum = 20
x = np.random.uniform(0, 2, num_of_sum)
noise = np.random.normal(0, 0.2, num_of_sum)
t = 3*x+noise

#initial_setting
w = 1
learning_rate = 0.004
E_save = []
w_itr = []

#iteration
num_of_itr = 25
for i in range(num_of_itr):
    y = w*x
    E = np.sum((y-t)**2)
    E_save = np.append(E_save, E)
    w_itr = np.append(w_itr, w)
    w = w - learning_rate*2*np.sum(x*(y-t))

    weight = []
    E_loss_weight = []

    num_of_w = 45
    resolution = 0.1

#plot_loss
for j in range(num_of_w):
    y_weight = (j*resolution)*x
    E_weight = np.sum((y_weight-t)**2)

    #strage data
    weight = np.append(weight, j*resolution)
    E_loss_weight = np.append(E_loss_weight, E_weight)

plt.figure()
plt.grid(True)
plt.title("LOSS FUNCTION")
plt.xlabel("Parameter(W)")
plt.ylabel("loss value")
plt.plot(weight, E_loss_weight, color='blue')
plt.plot(w_itr, E_save, 'o', color='red')
plt.savefig('figure_{}.png'.format(i))
plt.show()

#plot_output
X_line = np.linspace(0, 2, 5)
Y_line = w*X_line

plt.figure()
plt.grid(True)
plt.ylim(0,6)
plt.title("Deep Learning")
plt.xlabel("input(x)")
plt.ylabel("output(y)")
plt.plot(x, t, 'o')
plt.plot(X_line, Y_line, color='red')
plt.savefig('output')
plt.show()



