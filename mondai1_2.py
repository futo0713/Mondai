import numpy as np
import matplotlib.pyplot as plt

#dataset
num_of_sam = 30
x = np.random.uniform(0, 4, num_of_sam)
noise = np.random.normal(0, 0.5, num_of_sam)
t = 2*x+5+noise

#initial_setting
w = 0.2
b=0
learning_rate = 0.0018
E_save = []

#iteration
num_of_itr = 30
for i in range(num_of_itr):
    y = w*x+b
    E = np.sum((y-t)**2)
    E_save = np.append(E_save, E)

    w = w - learning_rate*2*np.sum(x*(y-t))
    b = b - learning_rate*2*np.sum(y-t)

#plot
X_line = np.linspace(0, 4, 5)
Y_line = w*X_line+b

#plot_output
plt.figure()
plt.grid(True)
plt.ylim(0,15)
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
plt.show()

