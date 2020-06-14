import xgboost
import pickle
import numpy as np
import gc
import pandas as pd

from bayes_opt import BayesianOptimization

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, precision_score

from general_functions import create_balanced_dataset


file = open("dataset/zone_4.pickle", "rb")
zone_4 = pickle.load(file)
file = None

file = open("dataset/zone_7.pickle", "rb")
zone_7 = pickle.load(file)
file = None

# zone_7_resampled = create_balanced_dataset(["dataset/zone_7.pickle"])

# with open("dataset/zone_7_resampled.pickle", "wb") as file:
#     pickle.dump(zone_7_resampled, file)

# zone_4_resampled = create_balanced_dataset(["dataset/zone_4.pickle"])

# with open("dataset/zone_4_resampled.pickle", "wb") as file:
#     pickle.dump(zone_4_resampled, file)

with open("dataset/zone_4_resampled.pickle", "rb") as file:
    zone_4_resampled = pickle.load(file)

with open("dataset/zone_7_resampled.pickle", "rb") as file:
    zone_7_resampled = pickle.load(file)
    
experiment_arr = [(zone_4_resampled, zone_7), (zone_7_resampled, zone_4)]

most_important_features = zone_4.columns.tolist()[1:]

def optim_function(learning_rate=.1,
                   n_estimators=100,
                   max_depth=5,
                   min_child_weight=1,
                   gamma=0,
                   subsample=.8,
                   colsample_bytree=.8,
                   scale_pos_weight=2,
                   reg_alpha=0,
                   reg_lambda=0):
    
    max_depth = int(max_depth)
    min_child_weight = int(min_child_weight)
    n_estimators = int(n_estimators)
    
    y_test_all = np.zeros((2, len(experiment_arr[0][1]))).astype(np.int8)
    pred_all = np.zeros((2, len(experiment_arr[0][1]))).astype(np.int8)
    
    for i, (training_dataset, test_dataset) in enumerate(experiment_arr):
        X_train = np.array(training_dataset.filter(items=most_important_features).loc[:, training_dataset.filter(items=most_important_features).columns != "label_3m"]).astype(np.float32)
        y_train = np.array(training_dataset["label_3m"]).astype(np.int8)
        
        training_dataset = None
        gc.collect()
        
        X_test = np.array(test_dataset.filter(items=most_important_features).loc[:, test_dataset.filter(items=most_important_features).columns != "label_3m"]).astype(np.float32)
        y_test = np.array(test_dataset["label_3m"]).astype(np.int8)
        
        test_dataset = None
        gc.collect()
        
        clf = xgboost.sklearn.XGBClassifier(max_depth=int(max_depth),
                                            learning_rate=learning_rate,
                                            n_estimators=n_estimators,
                                            gamma=gamma,
                                            min_child_weight=int(min_child_weight),
                                            subsample=subsample,
                                            colsample_bytree=colsample_bytree,
                                            scale_pos_weight=scale_pos_weight,
                                            reg_alpha=reg_alpha,
                                            reg_lambda=reg_lambda,
                                            tree_method="gpu_hist",
                                            seed=41,
                                            gpu_id=0,
                                            **{"predictor": "gpu_predictor"}
                                           )

        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_test)
        
        pred = np.array(pred).astype(np.int8)
        y_test = np.array(y_test).astype(np.int8)
        
        pred_all[i] = pred
        y_test_all[i] = y_test
        
    pred_all = pred_all.reshape(-1)
    y_test_all = y_test_all.reshape(-1)
    
    kappa = cohen_kappa_score(np.array(y_test_all), np.array(pred_all))
    return kappa

pbounds = {"learning_rate": (1e-4, 1e0),
           "n_estimators": (50.0, 500.0),
           "gamma": (0.0, 1.0),
           "min_child_weight": (1e-3, 30.0),
           "subsample": (.2, 1.0),
           "colsample_bytree": (.2, 1.0),
           "scale_pos_weight": (1.0, 4.0),
           "max_depth": (3.0, 30),
           "reg_alpha": (0.0, 1e-1),
           "reg_lambda": (0.0, 1e-1)
          }

optimizer = BayesianOptimization(
    f=optim_function,
    pbounds=pbounds,
    random_state=1,
    verbose=2,
)

try:
    optimizer.maximize(n_iter=120)

except:
    print(optimizer.max)

print(optimizer.max)