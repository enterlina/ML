from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
from matplotlib import pyplot
from sklearn import metrics
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk import RegexpTokenizer

nltk.download('stopwords')


def convert(df):
    df_label = []
    for i in range(len(df)):
        if df[i][0] == 'ham':
            df_label.append(0)
        elif df[i][0] == 'spam':
            df_label.append(1)
    return df_label





def get_dictionary(text):
    # text=convert_text(text)
    text = np.array_str(text)
    ps = PorterStemmer()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = text.lower()
    words = RegexpTokenizer(r'\w+').tokenize(text)
    return [ps.stem(word) for word in words]


def class_prob(y):
    y_dict = collections.Counter(y)
    class_prob = np.ones(2)
    for i in range(0, 2):
        class_prob[i] = y_dict[i] / y.shape[0]

    return class_prob


def get_word_num(X, y, word_dict):
    total_word_spam = 0
    total_word_ham = 0
    word_dict = ''.join(map(str, word_dict))
    total_word_dict = len(word_dict.split())
    for i in range(len(y)):
        if y[i] == 0:
            word_ham = ''.join(map(str, X[i]))
            ham_counts = len(word_ham.split())
            total_word_ham += ham_counts
        if y[i] == 1:
            word_spam = ''.join(map(str, X[i]))
            spam_counts = len(word_spam.split())
            total_word_spam += spam_counts

    return total_word_ham, total_word_spam, total_word_dict


def get_word_probability(X, y, word_dict):
    total_word_ham, total_word_spam, total_word_dict = get_word_num(X, y, word_dict)
    probability_dict = {}

    for word in word_dict:
        key = word
        count_zero = 0
        count_one = 0
        pfc = np.zeros(2)
        alpha = 0.0001
        for i in range(len(y)):
            row = ''.join(map(str, list(X[i])))
            for item in row.split():
                if y[i] == 0:
                    if item == word:
                        count_zero += 1
                if y[i] == 1:
                    if item == word:
                        count_one += 1
        pfc[0] = (count_zero + alpha) / (total_word_ham + total_word_dict)
        pfc[1] = (count_one + alpha) / (total_word_spam + total_word_dict)
        probability_dict[key] = pfc
    return probability_dict


def get_feature_probability(X, y, word_probability):
    predict_list = []
    pcf_list = []
    for i in range(len(y)):
        pcf = np.ones(2)
        row = ''.join(map(str, list(X[i])))
        for item in row.split():
            for word in word_probability:
                if item == word:
                    for j in range(0, 2):
                        pcf[j] = float(pcf[j] * word_probability[word][j])

        prediction = int(pcf.argmax())
        predict_list.append(prediction)
        pcf_list.append(pcf[1])
    return predict_list, pcf_list


df = pd.read_csv('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW8/smsspam.csv')
X = df.drop(['label'], axis=1)

df = df.values
df = np.array(df)
df_label = convert(df)

y = np.array(df_label)

X = np.array(X)
df_train, df_test, label_train, label_test = train_test_split(X, y, test_size=0.3, shuffle=True)

corpus_dict = get_dictionary(df_train)

word_probability = get_word_probability(df_train, label_train, corpus_dict)
print(word_probability)

predict_list, pcf_list = get_feature_probability(df_test, label_test, word_probability)


accuracy = accuracy_score(label_test, predict_list)
print('Accuracy', accuracy)
print('ROC-AUC', metrics.roc_auc_score(label_test, predict_list))

r_curve = metrics.roc_curve(label_test, pcf_list)

fpr, tpr, treshold = r_curve

pyplot.plot(fpr, tpr, marker='.')
pyplot.title('ROC-curve')
pyplot.show()
