import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


from utils import (
    read_data,
    get_shapelet,
    shapelet_category,
    plot_shapelet,
    train_model,
    targets_generation,
    eval_model,
    counterfacutal_generation,
    plot_cf_vs_ori,
    target_probability,
    cf_eval_res,
)

class Naive_Shapelet:
    def __init__(
        self,
        method='naive_shapelet',
        DS_name='ECG200',
        classifier=RandomForestClassifier(),
    ):
        self.method = method
        self.DS_name = DS_name
        self.classifier = classifier
        self.X_train, self.X_test, self.y_train, self.y_test = read_data(DS=self.DS_name)

    def generate_cf(self):
        len_ts = self.X_train.shape[1]
        classes = np.unique(self.y_test)
        nb_classes = len(classes)

        idx_shapelets = get_shapelet(self.X_train, self.y_train, len_ts)
        shapelets = [shapelet_category(self.y_train, idx_shapelets, i) for i in classes]

        shapelet_dict_list = [
            {'index': item[0], 'start': item[1], 'end': item[2], 'label': item[3]}
            for sublist in shapelets
            for item in sublist
        ]

        df = pd.DataFrame(shapelet_dict_list)
        df.to_csv(self.DS_name + '_' +'shapelet.csv', index=True)

        for i in range(nb_classes):
            plot_shapelet(self.X_train, shapelets, i, save_path= self.DS_name + '_' + str(i) +'_shapelet.png')

        model = train_model(self.classifier, self.X_train, self.y_train, self.X_test)

        y_pred = model.predict(self.X_test)
        y_preds = model.predict_proba(self.X_test)

        targets = targets_generation(y_preds)
        accuracy = eval_model(self.y_test, y_pred)

        counterfactual_examples = counterfacutal_generation(self.X_test, shapelets, targets, self.X_train)
        np.save(self.DS_name + '_' + self.method +'_cf.npy', counterfactual_examples)

        return counterfactual_examples, accuracy, model, targets

    def plot_comparison_res(self, index):
        counterfactual_examples = np.load(self.DS_name + '_' + self.method +'_cf.npy')
        plot_cf_vs_ori(self.DS_name, index, counterfactual_examples, save_path= self.DS_name + '_' + self.method +'_cfVori_fig.png')

    def save_res(self):
        counterfactual_examples, accuracy, model, targets = self.generate_cf()
        counter_res = model.predict_proba(counterfactual_examples)
        target_probs = target_probability(counter_res, targets)
        cf_eval_res(self.DS_name, self.method, model, accuracy, counterfactual_examples, targets, target_probs)

def main():
    naive_shapelet_instance = Naive_Shapelet()
    naive_shapelet_instance.save_res()
    naive_shapelet_instance.plot_comparison_res(index=-1)

# datasets_list = ['ECG200', 'Coffee', 'GunPoint', 'CBF', 'Chinatown']
# def main():
#     for dataset in datasets_list:
#         naive_shapelet_instance = Naive_Shapelet(DS_name=dataset)
#         naive_shapelet_instance.save_res()

if __name__ == "__main__":
    main()
