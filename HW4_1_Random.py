import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from statistics import mode
from sklearn import metrics
import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
import math
import statistics
from statistics import mode
import random


class Tree_Node:
    def __init__(self, left, right, feature, threshold, probability):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.probability = probability


def convert(df):
    # df_1 = df[:,0]

    df_label = []
    # print(df[:, 0])
    # print(len(df))
    for i in range(len(df)):
        if df[i][0] == 'M':
            df_label.append(1)
        elif df[i][0] == 'B':
            df_label.append(0)

    return df_label


def split(df, df_label):
    df_train, df_test, label_train, label_test = train_test_split(df, df_label, test_size=0.2, shuffle=True)
    return df_train, df_test, label_train, label_test


def counter(label):
    counter = 0
    if len(label) != 0:
        for i in range(len(label)):
            if label[i] == 1:
                counter += 1
        return counter / len(label)
    # else:
    #     return 0


def auc_classification(df, df_label, length):
    auc_dict = dict()
    for i in range(0, length):
        value = roc_auc_score(df_label, df[:, i])
        auc_dict[value] = i

    return auc_dict


def gini(p):
    return 1 - p ** 2 - (1 - p) ** 2


def get_ig(df, df_label, total, criteria):
    len1 = len(df)
    p = counter(df_label)
    if criteria == 'gini':
        return gini(p) * len1 / total


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


def calc_accuracy(tree_cart, df_test, label_test):
    for i in range(len(df)):
        predicted_label = get_label_from_tree(tree_cart, df_test[i])
        expected_label = label_test[i]
        if expected_label == 1:
            if predicted_label == 1:
                tp += 1
            else:
                fp += 1
        else:
            if predicted_label == 1:
                fn += 1
            else:
                tn += 1
    return (tp + tn) / (tp + fp + tn + fn)


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
    return df_left, label_left, df_right, label_right


def calc_p(column, label, criteria):
    count_a1 = 0
    count_a2 = 0
    count_b1 = 0
    count_b2 = 0
    ps_1 = 0
    ps_2 = 0
    ps_total = 0
    for i in range(len(column)):
        if column[i] > criteria:
            if label[i] == 1:
                count_a1 += 1
            else:
                count_a2 += 1

        else:
            if label[i] == 1:
                count_b1 += 1
            else:
                count_b2 += 1
    if (count_a1 == 0) and (count_a2 == 0):
        p_1 = 0

    else:
        p_1 = count_a1 / (count_a1 + count_a2)
        ps_1 = count_a1 + count_a2

    if (count_b1 == 0) and (count_b2 == 0):
        p_2 = 0
    else:
        p_2 = count_b1 / (count_b1 + count_b2)
        ps_2 = count_b1 + count_b2

    ps_total = count_a1 + count_a2 + count_b1 + count_b2

    return p_1, p_2, ps_1, ps_2, ps_total


def get_best_classific(df, df_label, total, classific, criteria):
    count = 0
    max_ig = 0
    best_classific = None
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

    return best_classific


def calc_criteria(df_test):
    df_average = []
    for i in range(len(df_test) - 1):
        df_average.append((df_test[i] + df_test[i + 1]) / 2)
    return df_average


def prob_label(tree, sample):
    threshold = tree.threshold
    if not threshold:
        return tree.label, tree.prob
    else:
        feature = tree.feature
        feature_val = sample[feature]
        if feature_val >= threshold:
            return prob_label(tree.right, sample)
        else:
            return prob_label(tree.left, sample)


def cart(df, df_label, length, depth):
    if counter(df_label) == 1:
        return Tree_Node(0, 0, 1, 0, 1)
    if counter(df_label) == 0:
        return Tree_Node(0, 0, 0, 0, 0)

    classific = get_classific(df, df_label)

    # depth_counter = 0
    max_depth = 3
    if depth < max_depth:
        feature, threshold = get_best_classific(df, df_label, total, classific, criteria)
        if (feature != None) and (threshold != None):
            df_left, label_left, df_right, label_right = split_classific(df, df_label, feature, threshold)
            depth_counter = depth + 1
            left_node = cart(df_left, label_left, length, depth_counter)
            right_node = cart(df_right, label_right, length, depth_counter)
            return Tree_Node(left_node, right_node, 0, threshold, 0)
        else:
            m_feature = mode(set(df_label))
            probabiliy = counter(df_label) / len(df_label)
            return Tree_Node(0, 0, m_feature, 0, probabiliy)
    else:
        m_feature = mode(set(df_label))
        probabiliy = counter(df_label) / len(df_label)
        return Tree_Node(0, 0, m_feature, 0, probabiliy)


def tree_bootstrap(df, df_label):
    index = np.random.choice(len(df), len(df))
    df_new = []
    df_label_new = []
    for i in range(len(df)):
        df_new.append(df[index[i]])
        df_label_new.append(df_label[index[i]])
    return np.asarray(df_new), np.asarray(df_label_new)


def random_forest(df, df_label, depth, tree_amount):
    total = len(df)
    tree_array = []

    for i in range(tree_amount):
        df_n, label_n = tree_bootstrap(df, df_label)
        tree_cart = cart(df_n, label_n, total, depth)
        print(tree_cart)
        tree_array.append(tree_cart)
    return tree_array


# cart(df, df_label, length, depth)

def get_classification_id(df, tree_cart):
    if (tree_cart.threshold != None) and (tree_cart.feature != None):
        if df[tree_cart.feature] > tree_cart.threshold:
            return get_labels(df, tree_cart.right)
        else:
            return get_labels(df, tree_cart.left)
    return tree_cart.label, tree_cart.probability


def roc_curve(df, label, tree_array):
    sum_of_probabilities = []
    for i in range(len(df)):
        prob = []
        for tree in tree_array:
            prob.append(get_labels(x[i], tree)[1])
        sum_of_probabilities.append(sum(prob))
    fpr, tpr, thresholds = metrics.roc_curve(label, np.array(sum_of_probabilities))
    auc_value = roc_auc_score(label, np.array(sum_of_probabilities))
    return fpr, tpr, auc_value


def plot_roc_curve(fpr, tpr):
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    pyplot.plot(fpr, tpr, marker='.', c='tab:orange')

    pyplot.show()


def print_auc(auc_value):
    print('Gini ')
    print('K: ', 3)
    print('N: ', 20)
    print('ROC-AUC: ', auc_value)


# df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW4/spam.csv')
df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW4/cancer2.csv')

df = df.values

df_label = convert(df)
df[:, 0] = df_label
length = len(df[1, :])
df_train, df_test, label_train, label_test = split(df, df_label)
p_1 = counter(label_train)
total = len(df[:, 0])
criteria = 'gini'
depth = 3
tree_amount = 20

print(get_ig(df, df_label, total, criteria))

classific = get_classific(df, df_label)


print(df_train, df_test, label_train, label_test)
df_train, label_train = np.array(df_train.drop(columns='label')), np.array(df_train.label)
tree_array = random_forest(df_train, label_train, criteria='gini')
,
df_test, label_test = np.array(df_test.drop(columns='label')), np.array(list(label_test.label))
fpr, tpr, auc_value = roc_curve(df_test, label_test, tree_array)

print_auc(auc_value)
plot_roc_curve(fpr, tpr)
print(list(df))
print(np.random.shuffle(list(df)))
