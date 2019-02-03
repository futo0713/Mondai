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

#iteration
num_of_itr = 25
for i in range(num_of_itr):
    y = w*x
    E = np.sum((y-t)**2)
    E_save = np.append(E_save, E)
    w = w - learning_rate*2*np.sum(x*(y-t))

#plot
X_line = np.linspace(0, 2, 5)
Y_line = w*X_line

#plot_output
plt.figure()
plt.grid(True)
plt.ylim(0,6)
plt.title("Deep Learning")
plt.xlabel("input(x)")
plt.ylabel("output(y)")
plt.plot(x, t, 'o')
plt.plot(X_line, Y_line, color='red')
plt.show()

#plot_loss
plt.figure()
plt.grid(True)
plt.title("LOSS FUNCTION")
plt.xlabel("Iteration number")
plt.ylabel("loss value")
plt.plot(E_save)
plt.savefig('loss')
plt.show()
