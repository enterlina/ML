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


# import warnings
# warnings.filterwarnings('ignore')

class Tree_Node:
    def __init__(self, left, right, feature, threshold, probability):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.probability = probability


def convert(df):
    df_1 = df.label

    df_label = df_1.values
    for i in range(len(df_label)):
        if df_label[i] == 'M':
            df_label[i] = 1
        elif df_label[i] == 'B':
            df_label[i] = 0

    df_label = list(df_label)
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
        # print(len(label), counter / len(label))
        return counter / len(label)
    # else:
    #     return 0


def auc_classification(df, label):
    auc_dict = dict()
    for i in range(1, df.shape[1]):
        value = roc_auc_score(label, df[df.columns[i]])
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
    # num_of_features = df.shape[1]
    features_auc = auc_classification(df, df_label)
    classific = []
    for auc, feature in features_auc.items():
        fpr, tpr, thresholds = metrics.roc_curve(df_label, df[df.columns[feature]])
        optimal = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal]
        classific.append((feature, optimal_threshold))
    return classific


def calc_accuracy(tree_cart, df_test, label_test):
    for i in range(df_test.shape[0]):
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
    for i in range(0, df.shape[0]):
        feature_value = df[df.columns[feature]]

        if feature_value[i] >= threshold:
            df_right.append(feature_value[i])
            label_right.append(df_label[i])
        else:
            df_left.append(feature_value[i])
            label_left.append(df_label[i])
    return df_left, label_left, df_right, label_right


def class_percent(df_label):
    num1 = 0
    num2 = 0
    for i in range(len(df_label)):
        if df_label[i] == 0:
            num1 += 1
        elif df_label[i] == 2:
            num2 += 1
    num1_per = num1 / len(df_label)
    num2_per = num2 / len(df_label)
    return num1_per, num2_per


def calc_criteria(df_test):
    df_average = []
    for i in range(len(df_test) - 1):
        df_average.append((df_test[i] + df_test[i + 1]) / 2)
    return df_average


def calc_best_ind(df_node, column, label, df_criteria, function):
    IG = []
    if function == 'gini':
        for i in range(len(df_criteria)):
            p_1, p_2, ps_1, ps_2, ps_total = calc_p(column, label, df_criteria[i])
            IG_local = df_node - (ps_1 / ps_total) * gini(p_1) - (ps_2 / ps_total) * gini(p_2)
            IG.append(IG_local)
    IG_ind = np.array(IG)
    ind = np.argpartition(IG_ind, -1)[-1:]
    return float(IG_ind[ind]), int(ind)


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


def get_best_classific(df, df_label, total, classific, criteria):
    count = 0
    max_ig = 0
    # total = df.shape[0]
    best_rule = None
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
            best_rule = (feature, optimal_threshold)
        print(best_rule)
    return best_rule


def cart(df, label, length, criteria, depth):
    rules = get_classific(df, label)
    print('rules', rules)
    depth_counter = 0
    max_depth = 3
    if depth < max_depth:
        feature, threshold = get_best_classific(df, label, total, rules, criteria)
        df_left, label_left, df_right, label_right = split_tree(df, label, feature_idx, threshold)
        depth_counter = depth + 1
        left_node = cart(df_left, label_left, length, criteria, depth_counter, depth)
        right_node = cart(df_right, label_right, length, criteria, new_depth, depth)
        return Tree_Node(left_node, right_node, feature, threshold)


def tree_bootstrap(df, label):
    index = np.random.choice(df.shape[0], df.shape[0])
    return df[index], label[index]


def random_forest(df, label, criteria='gini'):
    total = df.shape[0]
    tree_array = []
    for i in range(20):
        df, label = tree_bootstrap(df, label)
        tree_cart = cart(df, label, total, 'gini', 0)
        tree_array.append(tree_cart)
    return tree_array


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


# df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW3/spam.csv')
df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW3/cancer2.csv')
df_label = convert(df)
df['label'] = df_label

df_original = df.drop(columns='label')

df_train, df_test, label_train, label_test = split(df, df_label)

df_train, df_test, label_train, label_test = split(df, df_label)

df_train, label_train = np.array(df_train.drop(columns='label')), np.array(df_train.label)
tree_array = random_forest(df_train, label_train, criteria='gini')

df_test, label_test = np.array(df_test.drop(columns='label')), np.array(list(label_test.label))
fpr, tpr, auc_value = roc_curve(df_test, label_test, tree_array)

print_auc(auc_value)
plot_roc_curve(fpr, tpr)
