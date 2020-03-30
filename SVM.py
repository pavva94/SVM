
import numpy as np
import cvxopt

from kernels import linear_kernel, polynomial_kernel, gaussian_kernel


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
