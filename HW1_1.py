import csv
import numpy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

vectorcompare = np.asarray(
    ['16.99', '6.38', '112.8', '1001', '0.1184', '0.2776', '0.3001', '0.1471', '0.2419', '0.07871',
     '1.095', '0.9053', '8.589', '143.4', '0.006399', '0.04904', '0.05373', '0.01587', '0.03003',
     '0.006193', '25.38', '17.33', '144.6', '2019', '0.1622', '0.6656', '0.7119', '0.2654', '0.4601',
     '0.1189'], dtype=float)

compare_array = {}
cell_array = []
z = []

numbers = []
with open('/Users/alena_paliakova/Google Drive/!Bioinf_drive/02_MachinLearn/HW1/cancer2.csv') as csvfile:
    reader = csv.reader(csvfile)  # change contents to floats
    for row in reader:  # each row is a list
        cell_array.append(row)

c_array = np.asarray(cell_array, dtype=float)

for j in range(1, len(c_array)):
    z.append(np.linalg.norm(vectorcompare - c_array[j]))

for j in range(1, len(c_array)):
    numbers.append(j)
min_list = sorted(zip(numbers, z), key=lambda t: t[1])

k = 5
print('\nVector to compare \n', vectorcompare, '\n\n k value \n', k)

print('\nNearest cells: #N, Value')
for i in range(0, k):
    print(list(min_list[i]))
