import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_margin(X1_train, X2_train, svm):
    def f(x, w, b, c=0):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c
        return (-w[0] * x - b + c) / w[1]

    weight = svm.weight
    bias = svm.bias
    support_vector = svm.support_vector

    plt.title("Linear Kernel")
    plt.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    plt.plot(X2_train[:, 0], X2_train[:, 1], "bo")
    plt.scatter(support_vector[:, 0], support_vector[:, 1], s=100, c="g")

    # Plot the lines

    # w.x + b = 0
    a0 = -4
    a1 = f(a0, weight, bias)
    b0 = 4
    b1 = f(b0, weight, bias)
    plt.plot([a0, b0], [a1, b1], "k")

    # w.x + b = 1
    a0 = -4
    a1 = f(a0, weight, bias, 1)
    b0 = 4
    b1 = f(b0, weight, bias, 1)
    plt.plot([a0, b0], [a1, b1], "k--")

    # w.x + b = -1
    a0 = -4
    a1 = f(a0, weight, bias, -1)
    b0 = 4
    b1 = f(b0, weight, bias, -1)
    plt.plot([a0, b0], [a1, b1], "k--")

    plt.axis("tight")
    plt.show()


def plot_contour(X1_train, X2_train, svm):
    support_vector = svm.support_vector

    plt.title("Non Linear Kernel")
    plt.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    plt.plot(X2_train[:, 0], X2_train[:, 1], "bo")
    plt.scatter(support_vector[:, 0], support_vector[:, 1])

    min_bound = min(np.min(X1_train), np.min(X2_train)) - 1
    max_bound = max(np.max(X1_train), np.max(X2_train)) + 1

    X1, X2 = np.meshgrid(np.linspace(min_bound, max_bound, 50), np.linspace(min_bound, max_bound, 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = svm.project(X).reshape(X1.shape)

    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    plt.axis("tight")
    plt.show()


def plot_decision_regions(X, y, Xtest, Ytest, classifier, kernel, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_train_min, x1_train_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_train_min, x2_train_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1_test_min, x1_test_max = Xtest[:, 0].min() - 1, Xtest[:, 0].max() + 1
    x2_test_min, x2_test_max = Xtest[:, 1].min() - 1, Xtest[:, 1].max() + 1

    x1_min = min(x1_train_min, x1_test_min)
    x1_max = max(x1_train_max, x1_test_max)
    x2_min = min(x2_train_min, x2_test_min)
    x2_max = max(x2_train_max, x2_test_max)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
        plt.scatter(x=Xtest[Ytest == cl, 0], y=Xtest[Ytest == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[3], label=cl)

    plt.axis("tight")
    if kernel == "linear":
        plt.title("Linear Kernel")
    elif kernel == "non_linear":
        plt.title("Non Linear Kernel")
    plt.show()
