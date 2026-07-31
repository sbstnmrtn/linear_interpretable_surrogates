import surrogates as lr
import sys

param_seed = int(sys.argv[1])       # seed
param_sparsity = int(sys.argv[2])   # K
param_time = int(sys.argv[3])       # time limit
param_train = int(sys.argv[4])      # train
param_fs = int(sys.argv[5])         # FS
param_items = int(sys.argv[6])      # N
param_if = str(sys.argv[7])         # FI
param_glob_k = int(sys.argv[8])     # if global should be used: 1; else 0
param_col_k = int(sys.argv[9])      # if column should be used: 1; else 0
param_only_heu = int(sys.argv[10])  # if Lasso should be executed: 1; else 0


if param_only_heu == 0:
    lr.execute_exact(param_train, 100, param_items, param_fs, param_seed, param_sparsity, param_time, param_if, param_glob_k, param_col_k)
elif param_only_heu == 1:
    lr.execute_lasso(param_train, 100, param_items, param_fs, param_seed, param_sparsity, param_if, param_glob_k, param_col_k)
else:
    raise NotImplementedError

