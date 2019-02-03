import numpy as np
import matplotlib.pyplot as plt

num_of_sum = 40
x = np.random.uniform(0, 6, num_of_sum)
noise = np.random.normal(0, 0.1, num_of_sum)
t = np.sin(x)+noise

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

dimention = 20
W1 = np.random.randn(dimention, 1)
B1 = np.random.randn(dimention, 1)

W2 = np.random.randn(1, dimention)
B2 = np.random.randn(1, 1)

learning_rate = 0.002
E_save = []

num_of_itr = 3000
for i in range(num_of_itr):
    #forward propagation
    H = sigmoid(x*W1+B1)
    y = np.dot(W2,H)+B2
    E = np.sum((t-y)**2)
    E_save = np.append(E_save,E)

    #differential
    dW2 = 2*np.sum(H*(y-t),axis=1)
    dB2 = 2*np.sum(y-t)
    dW1 = 2*W2*np.sum(x*H*(1-H)*(y-t),axis=1)
    dB1 = 2*W2*np.sum(H*(1-H)*(y-t))

    #back propagation
    W1 = W1-learning_rate*dW1.T
    W2 = W2-learning_rate*dW2
    B1 = B1-learning_rate*dB1.T
    B2 = B2-learning_rate*dB2

#plot
X_line = np.linspace(0, 6, 200)
H_line = sigmoid(X_line*W1+B1)
Y_line = np.ravel(np.dot(W2,H_line)+B2)

#plot_output
plt.figure()
plt.grid(True)
plt.title("Deep Learning")
plt.xlabel("input(x)")
plt.ylabel("output(y)")
plt.plot(X_line, Y_line, color='red')
plt.plot(x, t, 'o')
plt.show()

#plot_loss
plt.figure()
plt.grid(True)
plt.title("LOSS FUNCTION")
plt.xlabel("Iteration number")
plt.ylabel("loss value")
plt.ylim(0,20)
plt.plot(E_save)
plt.show()
