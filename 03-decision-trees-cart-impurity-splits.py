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


class Tree_Node:
    def __init__(self, left, right, feature, threshold):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold


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
    # print(label)
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


def find_max_ind(auc_ind):
    auc_ind = np.array(auc_ind)
    ind = np.argpartition(auc_ind, -3)[-3:]
    return ind, auc_ind[ind]


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


def gini(p):
    return 1 - p ** 2 - (1 - p) ** 2


def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


def mis_error(p):
    return 1 - np.max([p, 1 - p])


def get_ig(df, df_label, total, criteria):
    len1 = len(df)
    p = counter(df_label)
    print('len', len1, 'total', total)
    if criteria == 'mis_error':
        return mis_error(p) * len1 / total
    elif criteria == 'entropy':
        return entropy(p) * len1 / total
    elif criteria == 'gini':
        return gini(p) * len1 / total


def get_classific(df, df_label):
    num_of_features = df.shape[1]
    features_auc = auc_classification(df, df_label)
    classific = []
    for auc, feature in features_auc.items():
        fpr, tpr, thresholds = metrics.roc_curve(df_label, df[df.columns[feature]])
        optimal = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal]
        classific.append((feature, optimal_threshold))
    return classific


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


def cart(df, label, length, criteria, depth):
    rules = get_classific(df, label)
    print('rules', rules)
    depth_counter = 0
    if depth < max_depth:
        feature, threshold = get_best_rule(df, label, total, rules, criteria)
        df_left, label_left, df_right, label_right = split_tree(df, label, feature_idx, threshold)
        depth_counter = depth + 1
        left_node = cart(df_left, label_left, length, criteria, depth_counter, depth)
        right_node = cart(df_right, label_right, length, criteria, new_depth, depth)
        return Tree_Node(left_node, right_node, feature, threshold)


def get_best_classific(df, df_label, classific, criteria):
    count = 0
    max_ig = 0
    total = df.shape[0]
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


# df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW3/spam.csv')
df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW3/cancer2.csv')
df_label = convert(df)

df['label'] = df_label

df_original = df.drop(columns='label')

df_train, df_test, label_train, label_test = split(df, df_label)

# df_train = pd.DataFrame(df_train)
# df_test = pd.DataFrame(df_test)
# print(df_test)
# p_1 = counter(label_train)
# p_2 = counter(label_test)

# criteria = ['gini', 'mis_error', 'entropy']

classific = get_classific(df, df_label)

total = df_train.shape[0]
df['label'] = df_label

df_original = df.drop(columns='label')

df_train, df_test, label_train, label_test = split(df, df_label)

criteria = 'gini'
accuracy_list = []
print('Gini impurity\n')
for i in range(1, 11):
    tree_cart = cart(df_train, label_train, df_train.shape[0], criteria, i)
    accuracy = calc_accuracy(tree_cart, df_test, label_test)
    accuracy_list.append(accuracy)
    print('k = ', i, 'accuracy = ', accuracy)

criteria = 'entropy'
accuracy_list = []
print('Entropy\n')
for i in range(1, 11):
    tree_cart = cart(df_train, label_train, df_train.shape[0], criteria, i)
    accuracy = calc_accuracy(tree_cart, df_test, label_test)
    accuracy_list.append(accuracy)
    print('k = ', i, 'accuracy = ', accuracy)

criteria = 'misclassification_error'
accuracy_list = []
print('Misclassification Error\n')
for i in range(1, 11):
    tree_cart = cart(df_train, label_train, df_train.shape[0], criteria, i)
    accuracy = calc_accuracy(tree_cart, df_test, label_test)
    accuracy_list.append(accuracy)
    print('k = ', i, 'accuracy = ', accuracy)




criteria = 'gini'
print('Gini impurity\n')
df_train, df_test, label_train, label_test = split(df, df_label)
tree_cart = cart(df_train, label_train, df_train.shape[0], criteria, i)
new_feature = []
    for i, sample in enumerate(df_test):
        pred_label, prob = prob_label(tree_cart, sample)
        new_feature.append(prob)
fpr, tpr, thresholds = skmetrics.roc_curve(label_test, new_feature)
print( criteria,'\n', 'k = \n', depth ,'auc = \n', skmetrics.roc_auc_score(label_test, new_feature))
pyplot.xlim(0, 1)
pyplot.ylim(0, 1)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.show()


criteria = 'entropy'
print('Entropy\n')
df_train, df_test, label_train, label_test = split(df, df_label)
tree_cart = cart(df_train, label_train, df_train.shape[0], criteria, i)
new_feature = []
    for i, sample in enumerate(df_test):
        pred_label, prob = prob_label(tree_cart, sample)
        new_feature.append(prob)
fpr, tpr, thresholds = skmetrics.roc_curve(label_test, new_feature)
print( criteria,'\n', 'k = \n', depth ,'auc = \n', skmetrics.roc_auc_score(label_test, new_feature))
pyplot.xlim(0, 1)
pyplot.ylim(0, 1)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.show()

criteria = 'misclassification_error'
print('Misclassification Error\n')
df_train, df_test, label_train, label_test = split(df, df_label)
tree_cart = cart(df_train, label_train, df_train.shape[0], criteria, i)
new_feature = []
    for i, sample in enumerate(df_test):
        pred_label, prob = prob_label(tree_cart, sample)
        new_feature.append(prob)
fpr, tpr, thresholds = skmetrics.roc_curve(label_test, new_feature)
print( criteria,'\n', 'k = \n', depth ,'auc = \n', skmetrics.roc_auc_score(label_test, new_feature))
pyplot.xlim(0, 1)
pyplot.ylim(0, 1)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.')
pyplot.show()
