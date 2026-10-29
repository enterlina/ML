# Machine Learning Algorithms — From-Scratch Implementations & Experiments

**Elena Dalboni da Rocha ([@enterlina](https://github.com/enterlina))** · [Portfolio](https://buildnxt.app) · [AI Engineering projects](https://github.com/enterlina/ai-engineering)

A curated collection of **29 Python scripts and Jupyter notebooks** from my bioinformatics machine-learning coursework. The focus is on understanding algorithms below the library API: implementing learning rules, distance functions, tree construction, statistical scoring, and optimization strategies, then exploring behavior through plots and evaluation metrics.

**Technical focus:** supervised learning · unsupervised learning · model evaluation · optimization · numerical computing · neural networks

> **Portfolio note:** These are historical educational/research exercises, not deployed production services. Some implementations are experimental or incomplete. The repository preserves original code, including known limitations, so the work is auditable rather than implying benchmarks or reliability that have not been verified.

## Projects and source code

### 01 · Nearest-neighbor methods

Distance-based classification on cancer and spam tables, with KD-tree queries, k-nearest neighbors, radius neighbors, leave-one-out-style error calculations, and a MinMax-scaled variant.

- [`01-nearest-neighbors-knn-radius-classification.py`](01-nearest-neighbors-knn-radius-classification.py) — nearest neighbors knn radius classification.
- [`01-nearest-neighbors-feature-scaling.py`](01-nearest-neighbors-feature-scaling.py) — nearest neighbors feature scaling.

### 02 · Unsupervised clustering

Explorations of k-means centroid updates, agglomerative nearest-pair merging, and DBSCAN-style density neighborhoods. Cluster quality is examined with the Dunn index and scatterplots.

- [`02-clustering-kmeans-dunn-index.ipynb`](02-clustering-kmeans-dunn-index.ipynb) — clustering kmeans dunn index.
- [`02-clustering-kmeans-cancer-data.py`](02-clustering-kmeans-cancer-data.py) — clustering kmeans cancer data.
- [`02-clustering-agglomerative-nearest-pairs.py`](02-clustering-agglomerative-nearest-pairs.py) — clustering agglomerative nearest pairs.
- [`02-clustering-dbscan-density-based.ipynb`](02-clustering-dbscan-density-based.ipynb) — clustering dbscan density based.

### 03 · Decision trees and classification metrics

An experimental CART-style decision tree with candidate feature thresholds and impurity criteria; separate ROC curves and AUC-based feature ranking for cancer and spam datasets.

- [`03-decision-trees-cart-impurity-splits.py`](03-decision-trees-cart-impurity-splits.py) — decision trees cart impurity splits.
- [`03-model-evaluation-roc-auc-feature-ranking.ipynb`](03-model-evaluation-roc-auc-feature-ranking.ipynb) — model evaluation roc auc feature ranking.

### 04 · Bagged tree ensembles

Bootstrap sampling for multiple hand-built decision trees, aggregate prediction scores, ROC/AUC visualization, and a stratified cross-validation grid over tree counts and depths.

- [`04-ensembles-bootstrap-decision-trees.py`](04-ensembles-bootstrap-decision-trees.py) — ensembles bootstrap decision trees.
- [`04-ensembles-cross-validated-tree-parameters.py`](04-ensembles-cross-validated-tree-parameters.py) — ensembles cross validated tree parameters.

### 05 · Perceptron learning

A perceptron weight-update loop with polynomial feature expansion and plotted nonlinear decision boundaries on a two-dimensional synthetic dataset.

- [`05-perceptron-polynomial-decision-boundaries.py`](05-perceptron-polynomial-decision-boundaries.py) — perceptron polynomial decision boundaries.

### 06 · Logistic regression and image classification

Hand-written logistic-regression loss, gradient updates, mini-batches and learning curves; MNIST/notMNIST image loading and normalization; Keras convolutional models comparing activations and character recognition.

- [`06-logistic-regression-minibatch-gradient-descent.py`](06-logistic-regression-minibatch-gradient-descent.py) — logistic regression minibatch gradient descent.
- [`06-image-preprocessing-mnist-notmnist.ipynb`](06-image-preprocessing-mnist-notmnist.ipynb) — image preprocessing mnist notmnist.
- [`06-cnn-mnist-activation-comparison.py`](06-cnn-mnist-activation-comparison.py) — cnn mnist activation comparison.
- [`06-cnn-notmnist-character-classification.py`](06-cnn-notmnist-character-classification.py) — cnn notmnist character classification.

### 07 · Support vector machines

Linear, polynomial and RBF SVM kernel decision boundaries; a separate comparison between linear SVM and a scikit-learn random forest on a cancer-classification exercise. Scripts and notebooks are preserved as companion formats.

- [`07-svm-kernel-decision-boundaries.ipynb`](07-svm-kernel-decision-boundaries.ipynb) — svm kernel decision boundaries.
- [`07-svm-kernel-decision-boundaries.py`](07-svm-kernel-decision-boundaries.py) — svm kernel decision boundaries.
- [`07-svm-vs-random-forest-classification.ipynb`](07-svm-vs-random-forest-classification.ipynb) — svm vs random forest classification.
- [`07-svm-vs-random-forest-classification.py`](07-svm-vs-random-forest-classification.py) — svm vs random forest classification.

### 08 · Probabilistic classification and regularized regression

Gaussian Naive Bayes from class priors and feature distributions; tokenization/stemming and word probabilities for SMS spam; polynomial least squares, Ridge (L2) and Lasso (L1) with R² and coefficient-count inspection.

- [`08-naive-bayes-gaussian-classification.py`](08-naive-bayes-gaussian-classification.py) — naive bayes gaussian classification.
- [`08-naive-bayes-sms-text-classification.py`](08-naive-bayes-sms-text-classification.py) — naive bayes sms text classification.
- [`08-regression-polynomial-least-squares.py`](08-regression-polynomial-least-squares.py) — regression polynomial least squares.
- [`08-regression-ridge-regularization.py`](08-regression-ridge-regularization.py) — regression ridge regularization.
- [`08-regression-lasso-feature-selection.py`](08-regression-lasso-feature-selection.py) — regression lasso feature selection.

### 09 · Combinatorial optimization: Traveling Salesperson Problem

Route-length minimization experiments using Monte Carlo permutations, random swaps, pair-swap hill climbing and a genetic algorithm with selection, crossover, mutation and elitism. Two hill-climbing files contain identical source.

- [`09-tsp-monte-carlo-search.py`](09-tsp-monte-carlo-search.py) — tsp monte carlo search.
- [`09-tsp-random-walk-search.py`](09-tsp-random-walk-search.py) — tsp random walk search.
- [`09-tsp-hill-climbing.py`](09-tsp-hill-climbing.py) — tsp hill climbing.
- [`09-tsp-hill-climbing-alternate-copy.py`](09-tsp-hill-climbing-alternate-copy.py) — Same source as the hill-climbing implementation; retained for provenance..
- [`09-tsp-genetic-algorithm.py`](09-tsp-genetic-algorithm.py) — tsp genetic algorithm.

## Methods and engineering evidence

| Capability | Evidence in this repository |
| --- | --- |
| Algorithm implementation | k-NN/radius voting, perceptron updates, CART-style splits, logistic-gradient steps, Gaussian and text Naive Bayes, route-search heuristics |
| Model selection and evaluation | Accuracy, leave-one-out-style checks, ROC curves, AUC, Dunn index, R², stratified cross-validation |
| Feature engineering | MinMax and standard scaling, polynomial feature maps, text tokenization/stemming, image normalization |
| Data visualization | Decision boundaries, cluster scatterplots, ROC plots, regression fits and TSP routes |
| Deep learning | TensorFlow/Keras CNN exercises for MNIST and notMNIST |

## Reproducibility and limitations

- The original datasets (`cancer.csv`, `spam.csv`, `blobs.csv`, `mnist.csv`, `notMNIST_small`, `tsp.csv` and others) are **not committed**. Scripts reference historical absolute paths that must be replaced before running.
- The historical environment is not pinned; modern Python, pandas, scikit-learn or Keras versions may require compatibility work.
- Review the implementations before reuse: several contain logic errors, incomplete branches or evaluation issues. These files have been **renamed and documented, not silently repaired or rerun**.
- The experiment titles describe the **code's intended method**, not an independently verified successful result. No production deployment, accuracy benchmark or clinical claim is made.

## About the author

I'm an AI Product Designer and Design Engineer with a background in mathematical mechanics, bioinformatics and end-to-end product development. This archive documents my foundations in algorithmic thinking alongside my newer [AI engineering work](https://github.com/enterlina/ai-engineering) and [product portfolio](https://buildnxt.app).
