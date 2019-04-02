import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
import math
import statistics
from statistics import mode
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from statistics import mode
from sklearn import metrics
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold


class Tree_Node:
    def __init__(self, left, right, label, feature, threshold, probability):
        self.left = left
        self.right = right
        self.feature = label
        self.feature = feature
        self.threshold = threshold
        self.probability = probability


def convert(df):
    df_label = []
    for i in range(len(df)):
        if df[i][0] == 'M':
            df_label.append(1)
        elif df[i][0] == 'B':
            df_label.append(0)
    for i in range(len(df)):
        if df[i][0] == 1:
            df_label.append(1)
        elif df[i][0] == 0:
            df_label.append(0)

    return df_label


def split(df, df_label):
    df_train, df_test, label_train, label_test = train_test_split(df, df_label, test_size=0.2, shuffle=True)
    return np.array(df_train), np.array(df_test), np.array(label_train), np.array(label_test)


def counter(label):
    counter = 0
    if len(label) != 0:
        for i in range(len(label)):
            if label[i] == 1:
                counter += 1
        return counter / len(label)
    else:
        return 0


def gini(p):
    return 1 - p ** 2 - (1 - p) ** 2


def get_ig(df, df_label, total, criteria):
    len1 = len(df)
    p = counter(df_label)
    if criteria == 'gini':
        return gini(p) * len1 / total


def auc_classification(df, df_label, length):
    auc_dict = dict()
    for i in range(0, length):
        value = roc_auc_score(np.array(list(df_label)), np.array((list(np.array(df[:, i])))))
        auc_dict[value] = i

    return auc_dict


# calculate all tresholds for all features
def get_classific(df, df_label):
    length = len(df[1, :])
    features_auc = auc_classification(df, df_label, length)
    classific = []
    for auc, feature in features_auc.items():
        fpr, tpr, thresholds = metrics.roc_curve(df_label, df[:, feature])
        optimal = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal]
        classific.append((feature, optimal_threshold))
    return classific


def split_classific(df, df_label, feature, threshold):
    df_left, label_left, df_right, label_right = [], [], [], []
    for label_ind, feature_ar in enumerate(df):
        feature_value = feature_ar[feature]
        label = df_label[label_ind]
        if feature_value >= threshold:
            df_right.append(feature_ar)
            label_right.append(label)
        else:
            df_left.append(feature_ar)
            label_left.append(label)
    return np.array(df_left), np.array(label_left), np.array(df_right), np.array(label_right)


def get_best_classific(df, df_label, total, classific, criteria):
    count = 0
    max_ig = 0
    best_classific = 0
    for feature, optimal_threshold in classific:
        df_left, label_left, df_right, label_right = split_classific(df, df_label, feature, optimal_threshold)
        if len(label_left) == 0 or len(label_right) == 0:
            count += 1
            continue
        ig_node = get_ig(df, df_label, total, criteria)
        ig_left = get_ig(df_left, label_left, total, criteria)
        ig_right = get_ig(df_right, label_right, total, criteria)

        ig = ig_node - ig_left - ig_right
        if ig == max_ig:
            count += 1
        if ig > max_ig:
            max_ig = ig
            best_classific = (feature, optimal_threshold)

    return best_classific if count < len(classific) else (0, 0)


def calc_criteria(df_test):
    df_average = []
    for i in range(len(df_test) - 1):
        df_average.append((df_test[i] + df_test[i + 1]) / 2)
    return df_average


def get_label(df_label):
    counter_y = 0
    counter_n = 0
    for i in range(len(df_label)):
        if (df_label[i] == 1):
            counter_y += 1
        else:
            counter_y += 1

    if counter_y >= counter_n:
        return 1
    else:
        return 0


def prob_label(tree, df):
    if (tree.threshold != 0) and (tree.feature != 0):
        feature = tree.feature
        if df[feature] > tree.threshold:
            return prob_label(tree.right, df)
        else:
            return prob_label(tree.left, df)
    return tree.feature, tree.probability


def cart(df, df_label, total, depth, max_depth):
    if counter(df_label) == 1:
        return Tree_Node(0, 0, 1, 0, 0, 1)
    if counter(df_label) == 0:
        return Tree_Node(0, 0, 0, 0, 0, 0)

    classific = get_classific(df, df_label)
    criteria = 'gini'
    if len(np.unique(df_label)) == 1:
        return Tree_Node(0, 0, np.unique(df_label), 0, 0, float(np.unique(df_label)))
    if depth <= max_depth:
        feature, threshold = get_best_classific(df, df_label, total, classific, criteria)
        if (feature != 0) and (threshold != 0):
            df_left, label_left, df_right, label_right = split_classific(df, df_label, feature, threshold)
            depth_counter = depth + 1
            left_node = cart(df_left, label_left, total, depth_counter, max_depth)
            right_node = cart(df_right, label_right, total, depth_counter, max_depth)
            return Tree_Node(left_node, right_node, 0, feature, threshold, 0)

        else:

            probabiliy = counter(df_label)
            return Tree_Node(0, 0, get_label(df_label), 0, 0, probabiliy)
    else:

        probabiliy = counter(df_label)
        return Tree_Node(0, 0, get_label(df_label), 0, 0, probabiliy)


def tree_bootstrap(df, df_label):
    index = np.random.choice(len(df), len(df))
    df_new = []
    df_label_new = []
    for i in range(len(df)):
        df_new.append(df[index[i]])
        df_label_new.append(df_label[index[i]])
    return np.asarray(df_new), np.asarray(df_label_new)


def random_forest(df, df_label, tree_amount, max_depth):
    total = len(df)
    tree_array = []

    for i in range(tree_amount):
        df_n, label_n = tree_bootstrap(df, df_label)
        tree_cart = cart(df_n, label_n, total, 0, max_depth)
        tree_array.append(tree_cart)
    return tree_array


def get_roc_curve(df, df_label, tree_array):
    probabilities = []
    for i in range(len(df)):
        prob = []
        for tree in tree_array:
            prob.append(prob_label(tree, df[i])[1])
        probabilities.append(sum(prob))
    probabilities, df_label = prune(df_label, probabilities)
    fpr, tpr, thresholds = metrics.roc_curve(df_label, probabilities)
    auc_value = roc_auc_score(df_label, probabilities)
    return fpr, tpr, auc_value


def prune(df_label, probabilities):
    probabilities = np.array(probabilities)
    list_elem_b = np.isnan(probabilities)
    list_elem = np.argwhere(list_elem_b)
    # print('list_elem_b',list_elem_b)
    print(len(list_elem))
    while len(list_elem) > 0:
        label_temp = np.argwhere(list_elem)[0]
        # print('!!!!label_temp', label_temp)
        df_label = np.delete(df_label, int(label_temp))

        probabilities = np.delete(probabilities, int(label_temp))
    return probabilities, df_label


def optimized_parameters(df, df_label, depth_of_trees, number_of_trees):
    depth_of_trees = [2, 3, 5, 7, 10]
    number_of_trees = [5, 10, 20, 30, 50, 100]
    criteria = 'gini'
    best_depth, best_number = 0, 0
    best_fpr, best_tpr = 0, 0
    current_max = 0
    n = 3
    skf = StratifiedKFold(n_splits=n, shuffle=True)

    for train_index, validate_index in skf.split(df, df_label):
        for depth in depth_of_trees:
            for number in number_of_trees:
                tree_array = random_forest(df[train_index], df_label[train_index], number, depth)
                fpr, tpr, auc_value = get_roc_curve(df[validate_index], df_label[validate_index], tree_array)
                if auc_value > current_max:
                    current_max = float(auc_value)
                    best_depth, best_number = depth, number
                    best_fpr, best_tpr = fpr, tpr

    print('\nGini ')
    print('K: ', best_depth)
    print('N: ', best_number)
    print_auc(current_max)
    plot_roc_curve(best_fpr, best_tpr)


def plot_roc_curve(fpr, tpr):
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    pyplot.plot(fpr, tpr, marker='.', c='tab:orange')

    pyplot.show()


def print_auc(auc_value):
    print('ROC-AUC: ', auc_value)


df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW4/spam.csv')
# df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW4/cancer.csv')

df = df.values

df_label = convert(df)
df_label = np.array(df_label)
df = np.delete(df, 0, 1)
df = np.array(df)

depth_of_trees = [2, 3, 5, 7, 10]
number_of_trees = [5, 10, 20, 30, 50, 100]

optimized_parameters(df, df_label, depth_of_trees, number_of_trees)
