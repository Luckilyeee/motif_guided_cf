import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import preprocessing
from scipy.spatial import distance
from pyts.transformation import ShapeletTransform
import pyts.datasets
from sklearn.metrics import accuracy_score

from tslearn.utils import to_sklearn_dataset
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

def read_data(DS):
    # print(pyts.datasets.ucr_dataset_list())
    X_train, X_test, y_train, y_test = pyts.datasets.fetch_ucr_dataset(dataset=DS, return_X_y=True)
    y_train, y_test = label_encoder(y_train, y_test)
    return X_train, X_test, y_train, y_test

def label_encoder(training_labels, testing_labels):

    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((training_labels, testing_labels), axis=0))
    y_train = le.transform(training_labels)
    y_test = le.transform(testing_labels)

    return y_train, y_test

def get_shapelet(X_train, y_train, len_ts):
    a = int(len_ts * 0.3)
    b = int(len_ts * 0.5)
    c = int(len_ts * 0.7)
    st = ShapeletTransform(n_shapelets=100, window_sizes=[a, b, c],
                           random_state=42, sort=True)
    st.fit_transform(X_train, y_train)
    indices = pd.DataFrame(st.indices_)
    return indices

# category the shapelets by their label, res include 4 columns, the index, start index, end index, and the label
def shapelet_category(y_train, indices, label):
    labels = y_train[indices.iloc[:, 0]]
    labels = pd.DataFrame(labels)
    frames = [indices, labels]
    res = pd.concat(frames, axis=1)
    res.columns=["idx", "start_point", "end_point", "label"]
    res = res.groupby('label')
    res = res.get_group(label).head(1)
    res = np.array(res)
    return res

def plot_shapelet(X_train, shapelets, cls, save_path = None):
    plt.style.use('bmh')
    plt.plot(X_train[shapelets[cls][0][0]], )
    plt.plot(np.arange(shapelets[cls][0][1], shapelets[cls][0][2]),
             X_train[shapelets[cls][0][0], shapelets[cls][0][1]:shapelets[cls][0][2]], label="shapelet")

    plt.xlabel('Time', fontsize=12)
    plt.title('The shapelets for class ' + str(cls), fontsize=14)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def train_model(classifier, X_train, y_train, X_test):
  model = classifier
  model.fit(X_train, y_train)
  return model

def target(instance):
  target = np.argsort(instance)[-2:-1][0]
  return target

def targets_generation(y_preds):
  targets= []
  for i in y_preds:
    res = target(i)
    targets.append(res)
  return targets

def eval_model(y_test, y_pred):
  accuracy = accuracy_score(y_test, y_pred)
  return accuracy

def counterfacutal_generation(test_samples, shapelets, targets, X_train):
    counterfactual_examples = []
    for i in range(len(test_samples)):
        index, start, end = shapelets[targets[i]][0][0], shapelets[targets[i]][0][1], shapelets[targets[i]][0][2]
        test_samples[i][start:end] = X_train[index][start:end]
        counterfactual_examples.append(test_samples[i])
    return counterfactual_examples

def getmetrics(x1, x2):
    x1 = np.round(x1, 3)
    x2 = np.round(x2, 3)

    l = np.round(x1 - x2, 3)
    l1 = distance.cityblock(x1, x2)
    l2 = np.linalg.norm(x1 - x2)  # Correct usage of np.linalg.norm for one-dimensional arrays
    l_inf = distance.chebyshev(x1, x2)
    sparsity = (len(l) - np.count_nonzero(l)) / len(l)

    segnums = get_segmentsNumber(l)
    return l1, l2, l_inf, sparsity, segnums

def get_segmentsNumber(l4):
    flag, count = 0,0
    for i in range(len(l4)):
        if l4[i:i+1][0]!=0:
            flag=1
        if flag==1 and l4[i:i+1][0]==0:
            count= count+1
            flag=0
    return count

def check_fliplabel(counterfactual_examples, model, targets):
  counter_res = model.predict(counterfactual_examples)
  accuracy = eval_model(counter_res,targets)
  return accuracy


def plot_cf_vs_ori(DS, index, counterfactual_examples, save_path=None):
    _, X_test, _, _ = pyts.datasets.fetch_ucr_dataset(dataset=DS, return_X_y=True)
    plt.style.use('bmh')
    plt.plot(X_test[index], label='Original', color='magenta')
    plt.plot(counterfactual_examples[index], label='CF', ls='--', color='green')

    plt.xlabel('Time', fontsize=12)
    plt.title('CF vs Ori on index ' + str(index), fontsize=14)
    plt.legend()

    # Save the figure if a save_path is provided
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def target_probability(counter_res, targets):
    target_probs = []
    for i in range(len(counter_res)):
        target_prob = counter_res[i][targets[i]]
        target_probs.append(target_prob)
    return target_probs

def cf_eval_res(DS, method, model, accuracy, counterfactual_examples, targets, target_probs):
    X_train, X_test, _, _ = pyts.datasets.fetch_ucr_dataset(dataset=DS, return_X_y=True)

    # Calculate metrics for each instance
    res = []
    for i in range(len(X_test)):
        l1, l2, l_inf, sparsity, segnums = getmetrics(X_test[i], counterfactual_examples[i])
        res.append([l1, l2, l_inf, sparsity, segnums, target_probs[i]])

    # Convert the list to a NumPy array
    res_array = np.array(res)

    # Calculate mean and std values for each column
    mean_values = np.mean(res_array, axis=0)
    std_values = np.std(res_array, axis=0)

    # Create a DataFrame for mean and std with the same column names
    summary_df = pd.DataFrame({'mean': mean_values, 'std': std_values},
                              index=['l1', 'l2', 'l_inf', 'sparsity', 'segnums', 'target_probs'])

    # Save the summary DataFrame to the final CSV file
    summary_df.to_csv(DS + '_' + method + '_summary.csv', index=True)

    # Calculate flip label rate
    flip_rate = check_fliplabel(counterfactual_examples, model, targets)

    # Evaluate OOD using SVM and LOF
    OOD_svm, OOD_lof, mean_OOD_ifo = cf_ood(X_train, counterfactual_examples)

    # Create a DataFrame for additional information
    additional_info_df = pd.DataFrame({
        'Flip_Label_Rate': [flip_rate],
        'Model_Accuracy': [accuracy],
        'OOD_SVM': [OOD_svm],
        'OOD_LOF': [OOD_lof],
        'OOD_IFO': [mean_OOD_ifo],

    })

    # Save the additional information DataFrame to the same CSV file
    additional_info_df.to_csv(DS + '_' + method + '_summary.csv', mode='a', header=True, index=True)


def cf_ood(X_train, counterfactual_examples):

    # Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=int(np.sqrt(len(X_train))), novelty=True, metric='euclidean')
    lof.fit(to_sklearn_dataset(X_train))

    novelty_detection = lof.predict(to_sklearn_dataset(counterfactual_examples))

    ood= np.count_nonzero(novelty_detection == -1)
    OOD_lof = ood / len(counterfactual_examples)

    # One-Class SVM (OC-SVM)
    clf = OneClassSVM(gamma='scale', nu=0.02).fit(to_sklearn_dataset(X_train))

    novelty_detection = clf.predict(to_sklearn_dataset(counterfactual_examples))

    ood = np.count_nonzero(novelty_detection == -1)
    OOD_svm = ood/ len(counterfactual_examples)

    # Initialize a list to store OOD results for min_edit_cf
    OOD_ifo = []

    # Loop over different random seeds
    for seed in range(10):
        iforest = IsolationForest(random_state=seed).fit(to_sklearn_dataset(X_train))

        novelty_detection = iforest.predict(to_sklearn_dataset(counterfactual_examples))

        ood = np.count_nonzero(novelty_detection == -1)

        OOD_ifo.append((ood/ len(counterfactual_examples)))

    mean_OOD_ifo = np.mean(OOD_ifo)

    return OOD_svm, OOD_lof, mean_OOD_ifo


