import argparse
import numpy as np
from sklearn.model_selection import train_test_split  # Only for convenience

from datasets import gen_lin_separable_data, gen_non_lin_separable_data, gen_non_lin_separable_data2, read_dataset, \
    gen_non_lin_separable_data3, gen_non_lin_separable_data4, gen_non_lin_separable_data5, gen_lin_separable_overlap_data
from plots import plot_decision_regions
from SVM import SVM


def test_linear(id_test, dataset_path=None, dataset_name=None):
    if id_test == 1:
        X1, y1, X2, y2 = gen_lin_separable_data()
    elif id_test == 2:
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
    else:
        print("ID test not valid.")
        return -2

    X_train, X_test, y_train, y_test = \
        train_test_split(np.concatenate((X1, X2)), np.concatenate((y1, y2)), train_size=0.6)

    svm = SVM("linear")
    svm.fit(X_train, y_train)

    y_predict = svm.predict(X_test)
    correct = int(np.sum(y_predict == y_test))
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    # plot_margin(X_train[y_train == 1], X_train[y_train == -1], svm)  # OLD with margin and no color
    plot_decision_regions(X_train, y_train, svm, "linear")


def test_non_linear(id_test, kernel, dataset_path=None, dataset_name=None):
    if dataset_path is None and dataset_name is None:
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
    else:
        X, y = read_dataset(dataset_path, dataset_name)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, train_size=0.8)

    # types: polynomial, gaussian
    svm = SVM(kernel_type=kernel)
    svm.fit(X_train, y_train)

    y_predict = svm.predict(X_test)
    correct = int(np.sum(y_predict == y_test))
    print("Correct predictions: %d out of %d" % (correct, len(y_predict)))

    # plot_contour(X_train[y_train == 1], X_train[y_train == -1], svm)  # OLD with margin and no color
    plot_decision_regions(X_train, y_train, svm, "non_linear")


def main(test_type, id_test=None, kernel_type=None, dataset_path=None, dataset_name=None):
    if test_type is None or id_test is None or kernel_type is None:
        print("Why are you here?")
        return -1
    if test_type == "linear":
        test_linear(int(id_test), dataset_path, dataset_name)
    elif test_type == "non_linear":
        test_non_linear(int(id_test), kernel_type, dataset_path, dataset_name)
    else:
        print("Invalid test's type.")
        return -1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Support Vector Machine from Scratch")
    parser.add_argument("--test_type", help="Select test's type from: [linear, non_linear]", default="linear")
    parser.add_argument("--test_number", help="Insert ID for LINEAR dataset: [1, 2]."
                                              "Insert ID for NON LINEAR dataset: [1:RandomNonLinear, 2:XDataset, "
                                              "3:MoonDataset, 4:CirclesDataset, 6:IrisDataset]",
                        default=1, required=False)
    parser.add_argument("--kernel_type", help="[ONLY FOR NON LINEAR] Select kernel's type from: [polynomial, gaussian]",
                        default="polynomial")
    parser.add_argument("--dataset_path", help="Insert path of your own dataset", default=None, required=False)
    parser.add_argument("--dataset_name", help="Insert name of your own dataset", default=None, required=False)
    args = parser.parse_args()
    print(args)
    main(test_type=args.test_type, id_test=args.test_number, kernel_type=args.kernel_type,
         dataset_path=args.dataset_path, dataset_name=args.dataset_name)
