import json
import numpy as np
import pandas as pd

import surrogates as lr


def execute(no_train, seed, sparsity, global_k, column_k, alpha):
    # train a linear surrogate using Lasso for a given value of alpha and K
    no_test = 100
    no_fs = 4
    no_items = 16
    if global_k >= 0.5 and column_k >= 0.5:
        raise NotImplementedError
    tr, te = lr.make_instance(no_train, no_test, no_items, no_fs, seed, if_parameter="all")
    lr_predictor_lasso = lr.solve_lr_knapsack_heu_lasso(tr, alpha, sparsity_param=sparsity, global_k=global_k,column_k=column_k)
    obj_tr_lr_lasso = lr.evaluate_lr(tr, lr_predictor_lasso)
    return obj_tr_lr_lasso


# for all sparsity types, all values of K used for the experiments, iterate over different values of alpha
# store objective values on the training data
# for each combination of sparsity definition and K, save the best alpha in a dictionary and store it as json file
arr_res = np.zeros((195, 4))
counter_row = 0
# global
for val_k in [4,8,16,24]:
    for a in [0.5, 1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250, 500, 1000]:
        obj = 0
        for s in range(25):
            obj += execute(10, s, val_k, 1, 0, a)
        arr_res[counter_row, 0] = 1
        arr_res[counter_row, 1] = val_k
        arr_res[counter_row, 2] = a
        arr_res[counter_row, 3] = obj
        counter_row += 1
        print(1)
# row
for val_k in [1,2,4,6]:
    for a in [0.5, 1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250, 500, 1000]:
        obj = 0
        for s in range(25):
            obj += execute(10, s, val_k, 0, 0, a)
        arr_res[counter_row, 0] = 2
        arr_res[counter_row, 1] = val_k
        arr_res[counter_row, 2] = a
        arr_res[counter_row, 3] = obj
        counter_row += 1
# col
for val_k in [1,2,4,6]:
    for a in [0.5, 1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250, 500, 1000]:
        obj = 0
        for s in range(25):
            obj += execute(10, s, val_k, 0, 1, a)
        arr_res[counter_row, 0] = 3
        arr_res[counter_row, 1] = val_k
        arr_res[counter_row, 2] = a
        arr_res[counter_row, 3] = obj
        counter_row += 1

df = pd.DataFrame(arr_res)
df.columns = ['type_of_k (1=glob 2=row 3=col)', 'value_of_k', 'value_of_alpha', 'obj_tr']
df_glob = df[df['type_of_k (1=glob 2=row 3=col)'] == 1]
df_row = df[df['type_of_k (1=glob 2=row 3=col)'] == 2]
df_col = df[df['type_of_k (1=glob 2=row 3=col)'] == 3]
df_glob_sorted = df_glob.sort_values(['value_of_k', 'obj_tr'])
r_glob = df_glob_sorted.drop_duplicates('value_of_k', keep='last')
df_row_sorted = df_row.sort_values(['value_of_k', 'obj_tr'])
r_row = df_row_sorted.drop_duplicates('value_of_k', keep='last')
df_col_sorted = df_col.sort_values(['value_of_k', 'obj_tr'])
r_col = df_col_sorted.drop_duplicates('value_of_k', keep='last')
# df.to_csv("results_parameter_training_alpha_lasso.csv", index=False)
dict_names = {1:"glob", 2:"row", 3:"col"}
r_glob['type_of_k (1=glob 2=row 3=col)']=r_glob['type_of_k (1=glob 2=row 3=col)'].map(dict_names)
r_col['type_of_k (1=glob 2=row 3=col)']=r_col['type_of_k (1=glob 2=row 3=col)'].map(dict_names)
r_row['type_of_k (1=glob 2=row 3=col)']=r_row['type_of_k (1=glob 2=row 3=col)'].map(dict_names)
dict_glob = dict(zip(zip(r_glob['type_of_k (1=glob 2=row 3=col)'], r_glob['value_of_k']), r_glob['value_of_alpha']))
dict_col = dict(zip(zip(r_col['type_of_k (1=glob 2=row 3=col)'], r_col['value_of_k']), r_col['value_of_alpha']))
dict_row = dict(zip(zip(r_row['type_of_k (1=glob 2=row 3=col)'], r_row['value_of_k']), r_row['value_of_alpha']))
dict_all = dict_glob | dict_col | dict_row
dict_for_json = {str(k): v for k, v in dict_all.items()}
with open("dict_alpha_values.json", "w") as file:
    json.dump(dict_for_json, file)

