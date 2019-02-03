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
    plt.figure(figsize=(10,8),dpi=200)
    plt.subplots_adjust(left=0.16, right=0.85, bottom=0.18, top=0.85)
    plt.grid(True)
    plt.ylim(0,6)
    plt.title("Deep Learning", fontsize=18)
    plt.xlabel("input(x)", fontsize=14)
    plt.ylabel("output(y)", fontsize=14)
    plt.plot(x, t, 'o')
    plt.plot(X_line, Y_line, color='red')
    plt.savefig('figure_{}.png'.format(i))
    # plt.show()

#plot_output
plt.figure(figsize=(10,8),dpi=200)
plt.subplots_adjust(left=0.16, right=0.85, bottom=0.18, top=0.85)
plt.grid(True)
plt.ylim(0,6)
plt.title("Deep Learning", fontsize=18)
plt.xlabel("input(x)", fontsize=14)
plt.ylabel("output(y)", fontsize=14)
plt.plot(x, t, 'o')
plt.savefig('plain')
# plt.show()

#plot_loss
plt.figure(figsize=(10,8),dpi=200)
plt.subplots_adjust(left=0.16, right=0.85, bottom=0.18, top=0.85)
plt.grid(True)
plt.title("LOSS FUNCTION", fontsize=18)
plt.xlabel("Iteration number", fontsize=14)
plt.ylabel("loss value", fontsize=14)
plt.plot(E_save)
plt.savefig('loss')
# plt.show()
