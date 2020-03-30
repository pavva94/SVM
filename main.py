import argparse
import numpy as np
from sklearn.model_selection import train_test_split  # Only for convenience

from datasets import gen_lin_separable_data, gen_non_lin_separable_data, gen_non_lin_separable_data2, \
    gen_non_lin_separable_data3, gen_non_lin_separable_data4, gen_non_lin_separable_data5
from plots import plot_contour, plot_margin
from SVM import SVM


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
    parser = argparse.ArgumentParser(description="Support Vector Machine from Scratch")
    parser.add_argument("--test_type", help="Select test's type from: [linear, non_linear]", default="linear")
    parser.add_argument("--test_number", help="Insert ID for non linear dataset: [1:RandomNonLinear, 2:XDataset, "
                                              "3:MoonDataset, 4:CirclesDataset, 6:IrisDataset]",
                        default=1, required=False)
    args = parser.parse_args()
    print(args)
    main(test_type=args.test_type, id_test=args.test_number)
