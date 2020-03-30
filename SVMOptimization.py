import argparse
import numpy as np
import cvxopt
import matplotlib.pyplot as plt

# Only for convenience
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, load_iris


class SVM:
    def __init__(self, kernel_type):
        self.kernel = None
        self.alpha = None
        self.support_vector = None
        self.support_vector_class = None
        self.bias = None
        self.weight = None

        if kernel_type == "linear":
            self.kernel = linear_kernel
        elif kernel_type == "polynomial":
            self.kernel = polynomial_kernel
        elif kernel_type == "gaussian":
            self.kernel = gaussian_kernel
        else:
            print("Wrong or missing kernel's type.")
            return

    def fit(self, data, label):
        """
        # Starting from F(A) = sum(A) - 1/2 * sum[i](sum[j](Ai*Aj*yi*yj*K(xi,xj)))
        # we have to reach the form: 1/2*Zt*Q*z + cT*z
        # can be noted that we have a minus before the second part (- 1/2 * sum[i] ....)
        # So it's necessary to change the sign to: - F(A) = - sum(A) + 1/2 * sum[i](sum[j](Ai*Aj*yi*yj*K(xi,xj)))
        # Now we rewrite the formula as: 1/2*At*Y*K*Yt*A + (-It*A)
        # And so we have that:
        # K = kernel(Xi,Xj)
        # P = Y * K * Yt = Q
        # q = -It = cT
        # Z is our variable to optimize

        # The constrain are: L >= 0 and sum(L*Y) = 0
        # that need to corresponds to: Gx<=H and Ax=b
        # so we have to transform like this: -I*L<=0 and ...
        # G = -I
        # x = L
        # H = 0
        # A = Y
        # b = 0

        # Now we can use the quadratic programming solver of CVXOPT

        :param data: dataset
        :param label: label of dataset
        :return:
        """
        n_samples, n_features = data.shape

        # build the kernel matrix calculating for every point the correspond value of kernel
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(data[i], data[j])

        # Calculate every argument for the quadrati solver of CVXOPT
        P = cvxopt.matrix(np.outer(label, label) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(label, (1, n_samples))
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers as a contiguous array
        a = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers so check this out
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]

        # save the Lagrange multiplier only for support vector
        self.alpha = a[sv]
        # save the support vector
        self.support_vector = data[sv]
        # save support vector label
        self.support_vector_class = label[sv]

        # Formula: b = 1/len(alphas) * sum(Yi - sum(AiYiK(Xi,Xj)))
        self.bias = 0
        for n in range(len(self.alpha)):  # sum(Yi - sum(AiYiK(Xi,Xj)))
            self.bias += self.support_vector_class[n]  # + Yi
            self.bias -= np.sum(self.alpha * self.support_vector_class * K[ind[n], sv])  # - sum(AiYiK(Xi,Xj)
        self.bias /= len(self.alpha)  # 1/len(alphas)

        # Weight vector
        if self.kernel == linear_kernel:
            self.weight = np.zeros(n_features)
            for n in range(len(self.alpha)):
                self.weight += self.alpha[n] * self.support_vector_class[n] * self.support_vector[n]
        else:
            self.weight = None

    def project(self, data):
        if self.weight is not None:  # LINEAR pag 51 of slide
            return np.dot(data, self.weight) + self.bias
        else:  # NON LINEAR Paper
            y_predict = np.zeros(len(data))

            # Formula: f(x) = sign(sum(Yi*Ai*K(x,Xi)) + b)
            for i in range(len(data)):
                s = 0
                for a, svy, sv in zip(self.alpha, self.support_vector_class, self.support_vector):
                    s += a * svy * self.kernel(data[i], sv)
                y_predict[i] = s
            return y_predict + self.bias

    def predict(self, data):
        return np.sign(self.project(data))


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=5):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=3.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


def gen_lin_separable_data():
    # generate training data in the 2-d case
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


def gen_non_lin_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


def gen_non_lin_separable_data2():
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1).astype("float")
    return X_xor, y_xor


def gen_non_lin_separable_data3():
    X, y = make_moons(n_samples=100, noise=0)
    y = np.where(y, 1, -1).astype("float")
    return X, y.astype("float")


def gen_non_lin_separable_data4():
    X, y = make_circles(n_samples=100, noise=0)
    y = np.where(y, 1, -1).astype("float")
    return X, y.astype("float")


def gen_non_lin_separable_data5():
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    y = np.where(y, 1, -1).astype("float")
    return X, y.astype("float")


def gen_lin_separable_overlap_data():
    # generate training data in the 2-d case
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


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
    plt.scatter(support_vector[:, 0], support_vector[:, 1], cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    min_bound = min(np.min(X1_train), np.min(X2_train)) - 1
    max_bound = max(np.max(X1_train), np.max(X2_train)) + 1

    X1, X2 = np.meshgrid(np.linspace(min_bound, max_bound, 50), np.linspace(min_bound, max_bound, 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = svm.project(X).reshape(X1.shape)

    plt.contourf(X1, X2, Z, cmap=plt.cm.coolwarm, alpha=1)

    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

    plt.axis("tight")
    plt.show()


def test_linear():
    X1, y1, X2, y2 = gen_lin_separable_data()

    X_train, X_test, y_train, y_test = \
        train_test_split(np.concatenate((X1, X2)), np.concatenate((y1, y2)), train_size=0.6)

    svm = SVM("linear")
    svm.fit(X_train, y_train)

    y_predict = svm.predict(X_test)
    correct = int(np.sum(y_predict == y_test))
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    plot_margin(X_train[y_train == 1], X_train[y_train == -1], svm)


def test_non_linear(id_test):
    if id_test == 1:
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, X_test, y_train, y_test = \
            train_test_split(np.concatenate((X1, X2)), np.concatenate((y1, y2)), train_size=0.6)
    elif id_test == 2:
        X, y = gen_non_lin_separable_data2()
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, train_size=0.6)
    elif id_test == 3:
        X, y = gen_non_lin_separable_data3()
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, train_size=0.6)
    elif id_test == 4:
        X, y = gen_non_lin_separable_data4()
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, train_size=0.6)
    elif id_test == 5:
        X, y = gen_non_lin_separable_data5()
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, train_size=0.8)
    else:
        print("ID test not valid.")
        return -2

    # types: polynomial, gaussian
    svm = SVM("polynomial")
    svm.fit(X_train, y_train)

    y_predict = svm.predict(X_test)
    correct = int(np.sum(y_predict == y_test))
    print("Correct predictions: %d out of %d" % (correct, len(y_predict)))

    plot_contour(X_train[y_train == 1], X_train[y_train == -1], svm)


def main(test_type, id_test=None):
    if test_type == "linear":
        test_linear()
    elif test_type == "non_linear":
        test_non_linear(int(id_test))
    else:
        print("Invalid test's type.")
        return -1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM")
    parser.add_argument("--test_type", help="Select test's type from: [linear, non_linear]", default="linear")
    parser.add_argument("--test_number", help="Insert ID for non linear dataset: [1:RandomNonLinear, 2:XDataset, 3:MoonDataset, 4:CirclesDataset, 6:IrisDataset]",
                        default=1, required=False)
    args = parser.parse_args()
    main(test_type=args.test_type)
