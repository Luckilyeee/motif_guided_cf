import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from scipy.spatial import distance
from pyts.transformation import ShapeletTransform
import pyts.datasets
from sklearn.metrics import accuracy_score

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

def plot_shapelet(X_train, shapelets, cls):
    plt.style.use('bmh')
    plt.plot(X_train[shapelets[cls][0][0]], )
    plt.plot(np.arange(shapelets[cls][0][1], shapelets[cls][0][2]),
             X_train[shapelets[cls][0][0], shapelets[cls][0][1]:shapelets[cls][0][2]], label="shapelet")

    plt.xlabel('Time', fontsize=12)
    plt.title('The shapelets for class ' + str(cls), fontsize=14)
    plt.legend()
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


def getmetrics(x1,x2):
    x1 = [np.round(e, 3) for e in x1]
    x2 = [np.round(e, 3) for e in x2]

    l = [np.round(e1-e2,3) for e1,e2 in zip(x1,x2)]
    dist = distance.cityblock(x1,x2)
    sparsity = (len(l)-np.count_nonzero(l))/len(l)

    segnums = get_segmentsNumber(l)
    return dist, sparsity, segnums

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

def plot_cf_vs_ori(DS, index, counterfactual_examples):
    _, X_test, _, _ = pyts.datasets.fetch_ucr_dataset(dataset=DS, return_X_y=True)
    plt.style.use('bmh')
    plt.plot(X_test[index], label = 'Original', color='magenta')
    plt.plot(counterfactual_examples[index], label = 'CF', ls='--', color='green')

    plt.xlabel('Time', fontsize=12)
    plt.title('CF vs Ori on index ' + str(index), fontsize=14)
    plt.legend()
    plt.show()

def target_probability(counter_res, targets):
    target_probs = []
    for i in range(len(counter_res)):
        target_prob = counter_res[i][targets[i]]
        target_probs.append(target_prob)
    return target_probs

def cf_eval_res(DS, method, model, accuracy, counterfactual_examples, targets, target_probs):
    _, X_test, _, _ = pyts.datasets.fetch_ucr_dataset(dataset=DS, return_X_y=True)
    res = []
    for i in range(len(X_test)):
        dist, sparsity, segnums = getmetrics(X_test[i], counterfactual_examples[i])
        res.append(np.array([dist, sparsity, segnums, target_probs[i]]))
    res = pd.DataFrame(res)
    res.columns = ['dist', 'sparsity', 'segnums', 'target_probs']
    res.to_csv(DS + '_' + method + '.csv')

    res = check_fliplabel(counterfactual_examples, model, targets)
    print("Model classification accuracy ", accuracy)
    print("flip label rate " + str(res))
