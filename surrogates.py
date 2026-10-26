import json
import time
import numpy as np
from gurobipy import *
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


class Instance:
    def __init__(self, profits, weights, budgets, feat_matrix, grouping):
        N, FI = feat_matrix.shape
        _, n = profits.shape
        self.p = profits
        self.w = weights
        self.FS = range(len(grouping))
        self.FI = range(FI)
        self.N = range(N)
        self.n = range(n)
        self.B = budgets
        self.ifs = feat_matrix
        self.grp = grouping
        self.ifs_dict = {}

    def get_params(self):
        return self.ifs, self.B, self.FI, self.FS, self.grp, self.n, self.N, self.p, self.w


class LRPredictor:
    def __init__(self, beta, bias):
        self.beta = beta
        self.bias = bias


class DTPredictor:
    def __init__(self):
        self.depth = 2
        self.eps = 0.0
        self.feature1 = 0
        self.feature2 = 0
        self.feature3 = 0
        self.threshold1 = 0.0
        self.threshold2 = 0.0
        self.threshold3 = 0.0
        self.sol1 = []
        self.sol2 = []
        self.sol3 = []
        self.sol4 = []


def make_instance(no_train, no_test, no_items, no_fs, seed, if_parameter="all"):
    # create an instance
    if no_train > 250 or no_test > 250:
        raise ValueError

    no_base_scenarios = 3
    no_scenarios_all = 500
    np.random.seed(seed)

    weights_all = np.zeros((no_scenarios_all, no_items))
    profits_all = np.zeros((no_scenarios_all, no_items))
    budgets_all = np.zeros((no_scenarios_all))
    base_scenarios_pw = np.zeros((no_base_scenarios, no_fs))
    grouping = np.array_split(range(no_items), no_fs)
    dict_it_to_grp = {item: group_idx for group_idx, group in enumerate(grouping) for item in group}

    for fs in range(no_fs):
        for bs in range(no_base_scenarios):
            pw_ratio = np.random.random_sample()*3.5+0.5
            base_scenarios_pw[bs,fs] = pw_ratio

    for sc in range(no_scenarios_all):
        idx_base = np.random.randint(0, no_base_scenarios)
        for it in range(no_items):
            profit_item = np.random.random_sample()*20+10
            weight_item_not_pert = profit_item / base_scenarios_pw[idx_base, dict_it_to_grp[it]]
            weight_item_pert = (1+np.random.random_sample()*0.1-0.05)*weight_item_not_pert
            profits_all[sc, it] = profit_item
            weights_all[sc, it] = weight_item_pert

    # scale weights s.t. sum per row is 100
    weights_all_scaled = weights_all[:, :no_items] / weights_all[:, :no_items].sum(axis=1, keepdims=True) * 100

    # set all budgets to 1/2 sum w (=50)
    for i in range(no_scenarios_all):
        budgets_all[i] = 50

    # splits items into groups
    grouping = np.array_split(range(no_items), no_fs)

    # training and test data
    profits_train = profits_all[:no_train,:]
    profits_test = profits_all[-no_test:,:]
    weights_train = weights_all_scaled[:no_train,:]
    weights_test = weights_all_scaled[-no_test:,:]
    budgets_train = budgets_all[:no_train]
    budgets_test = budgets_all[-no_test:]

    # calculate instance features
    if if_parameter == "all":
        instance_feats_train = np.zeros((no_train, 10 * no_fs + 2 * no_items))
        instance_feats_test = np.zeros((no_test, 10 * no_fs + 2 * no_items))
    elif if_parameter == "individual":
        instance_feats_train = np.zeros((no_train, 2 * no_items))
        instance_feats_test = np.zeros((no_test, 2 * no_items))
    elif if_parameter == "grouped":
        instance_feats_train = np.zeros((no_train, 10 * no_fs))
        instance_feats_test = np.zeros((no_test, 10 * no_fs))
    else:
        raise NotImplementedError

    # create a dictionary to analyze instance features used in the model
    dict_instance_feats = {}

    if if_parameter == "all" or if_parameter == "grouped":
        for fs in range(no_fs):
            for i in range(no_train):
                instance_feats_train[i, fs] = sum(profits_train[i,j] for j in grouping[fs])/sum(weights_train[i,j] for j in grouping[fs])
                instance_feats_train[i, 1*no_fs + fs] = sum(weights_train[i, j] for j in grouping[fs])
                instance_feats_train[i, 2*no_fs + fs] = min(weights_train[i, j] for j in grouping[fs])
                instance_feats_train[i, 3*no_fs + fs] = max(weights_train[i, j] for j in grouping[fs])
                instance_feats_train[i, 4*no_fs + fs] = sum(profits_train[i, j] for j in grouping[fs])
                instance_feats_train[i, 5*no_fs + fs] = min(profits_train[i, j] for j in grouping[fs])
                instance_feats_train[i, 6*no_fs + fs] = max(profits_train[i, j] for j in grouping[fs])
                instance_feats_train[i, 7*no_fs + fs] = sum(weights_train[i, j] for j in grouping[fs])/sum(weights_train[i, j] for j in range(no_items))
                instance_feats_train[i, 8*no_fs + fs] = sum(profits_train[i, j] for j in grouping[fs])/sum(profits_train[i, j] for j in range(no_items))
                # instance_feats_train[i, 9*no_fs + fs] = instance_feats_train[i, 1*no_fs + fs]/(sum(profits_train[i,j] for j in range(no_items))/sum(weights_train[i,j] for j in range(no_items)))
                instance_feats_train[i, 9*no_fs + fs] = instance_feats_train[i, fs]/(sum(profits_train[i,j] for j in range(no_items))/sum(weights_train[i,j] for j in range(no_items)))
            for i in range(no_test):
                instance_feats_test[i, fs] = sum(profits_test[i,j] for j in grouping[fs])/sum(weights_test[i,j] for j in grouping[fs])
                instance_feats_test[i, 1*no_fs + fs] = sum(weights_test[i, j] for j in grouping[fs])
                instance_feats_test[i, 2*no_fs + fs] = min(weights_test[i, j] for j in grouping[fs])
                instance_feats_test[i, 3*no_fs + fs] = max(weights_test[i, j] for j in grouping[fs])
                instance_feats_test[i, 4*no_fs + fs] = sum(profits_test[i, j] for j in grouping[fs])
                instance_feats_test[i, 5*no_fs + fs] = min(profits_test[i, j] for j in grouping[fs])
                instance_feats_test[i, 6*no_fs + fs] = max(profits_test[i, j] for j in grouping[fs])
                instance_feats_test[i, 7*no_fs + fs] = sum(weights_test[i, j] for j in grouping[fs])/sum(weights_test[i, j] for j in range(no_items))
                instance_feats_test[i, 8*no_fs + fs] = sum(profits_test[i, j] for j in grouping[fs])/sum(profits_test[i, j] for j in range(no_items))
                # instance_feats_test[i, 9*no_fs + fs] = instance_feats_test[i, 1*no_fs + fs]/(sum(profits_test[i,j] for j in range(no_items))/sum(weights_test[i,j] for j in range(no_items)))
                instance_feats_test[i, 9*no_fs + fs] = instance_feats_test[i, fs]/(sum(profits_test[i,j] for j in range(no_items))/sum(weights_test[i,j] for j in range(no_items)))
            dict_instance_feats[fs], dict_instance_feats[1*no_fs + fs] = "p/w", "sum_w"
            dict_instance_feats[2*no_fs + fs], dict_instance_feats[3*no_fs + fs] = "min_w", "max_w"
            dict_instance_feats[4*no_fs + fs], dict_instance_feats[5*no_fs + fs] = "sum_p", "min_p"
            dict_instance_feats[6*no_fs + fs] = "max_p"
            dict_instance_feats[7*no_fs + fs] = "w_vs"
            dict_instance_feats[8*no_fs + fs] = "p_vs"
            dict_instance_feats[9*no_fs + fs] = "pw_vs"
    if if_parameter == "all":
        for i in range(no_train):
            instance_feats_train[i, 9*no_fs + no_fs:9*no_fs + no_fs+no_items] = weights_train[i,:]
            instance_feats_train[i, 9*no_fs + no_fs+no_items:9*no_fs + no_fs+2*no_items] = profits_train[i, :]
        for i in range(no_test):
            instance_feats_test[i, 9*no_fs + no_fs:9*no_fs + no_fs+no_items] = weights_test[i,:]
            instance_feats_test[i, 9*no_fs + no_fs+no_items:9*no_fs + no_fs+2*no_items] = profits_test[i, :]
        for j in range(no_items):
            dict_instance_feats[9 * no_fs + no_fs + j] = "w_j"
            dict_instance_feats[9 * no_fs + no_fs + no_items + j] = "p_j"
    elif if_parameter == "individual":
        for i in range(no_train):
            instance_feats_train[i, :no_items] = weights_train[i,:]
            instance_feats_train[i, no_items:2*no_items] = profits_train[i, :]
        for i in range(no_test):
            instance_feats_test[i, :no_items] = weights_test[i,:]
            instance_feats_test[i, no_items:2*no_items] = profits_test[i, :]
        for j in range(no_items):
            dict_instance_feats[j] = "w_j"
            dict_instance_feats[no_items + j] = "p_j"

    instance_train = Instance(profits_train, weights_train, budgets_train, instance_feats_train, grouping)
    instance_test = Instance(profits_test, weights_test, budgets_test, instance_feats_test, grouping)
    instance_train.ifs_dict = dict_instance_feats
    instance_test.ifs_dict = dict_instance_feats

    return instance_train, instance_test


# +++++ LINEAR REGRESSION +++++


def solve_lr_knapsack(instance, time_limit, sparsity_param=2, global_k=0, column_k=0):
    # solve the exact approach
    feat_matrix, budgets, FI, FS, grp, n, N, p, w = instance.get_params()

    m = Model()
    m.Params.timeLimit = time_limit
    m.Params.OutputFlag = 0
    m.Params.Threads = 1

    beta = m.addVars(FS, FI, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="beta")
    bias = m.addVars(FS, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="bias")
    b = m.addVars(N, FS, vtype=GRB.CONTINUOUS, ub=1, name="b")
    x = m.addVars(N, n, vtype=GRB.BINARY, name="x")
    s = m.addVars(FS, FI, vtype=GRB.BINARY, name="s")
    # auxiliary variables for column wise sparsity
    if column_k >= 0.5:
        aux = m.addVars(FI, vtype=GRB.BINARY, name="aux")

    # we make use of M=10 to ensure interpretability and avoid numerical issues
    M = 10

    m.setObjective(quicksum(p[i, j] * x[i, j] for i in N for j in n), GRB.MAXIMIZE)

    # regression + knapsack
    m.addConstrs(quicksum(beta[fs, fi] * feat_matrix[i, fi] for fi in FI) + bias[fs] == b[i, fs] for i in N for fs in FS)
    m.addConstrs(quicksum(b[i, fs] for fs in FS) <= 1 for i in N)
    m.addConstrs(quicksum(x[i, j] * w[i, j] for j in grp[fs]) <= b[i, fs] * budgets[i] for i in N for fs in FS)

    # sparsity
    m.addConstrs(-M * s[fs, fi] <= beta[fs, fi] for fs in FS for fi in FI)
    m.addConstrs(beta[fs, fi] <= M * s[fs, fi] for fs in FS for fi in FI)

    # auxiliary variables for global sparsity
    if global_k >= 0.5:
        m.addConstr(quicksum(s[fs, fi] for fi in FI for fs in FS) <= sparsity_param)
    # auxiliary variables for column wise sparsity
    elif column_k >= 0.5:
        m.addConstrs(quicksum(s[fs, fi] for fs in FS) <= aux[fi] * len(FS) for fi in FI)
        m.addConstr(quicksum(aux[fi] for fi in FI) <= sparsity_param)
    # auxiliary variables for row wise sparsity
    else:
        m.addConstrs(quicksum(s[fs, fi] for fi in FI) <= sparsity_param for fs in FS)

    m.optimize()

    beta_out = np.zeros((len(FS), len(FI)))
    bias_out = np.zeros((len(FS)))
    for fs in FS:
        bias_out[fs] = bias[fs].X
        for fi in FI:
            beta_out[fs, fi] = beta[fs, fi].X
    predictor = LRPredictor(beta_out, bias_out)

    return m.ObjVal, m.MIPGap, predictor


def pp_heu(instance):
    # do preprocessing fpr the Lasso heuristic
    feat_matrix, budgets, FI, FS, grp, n, N, p, w = instance.get_params()

    # solve knapsack problem for each scenario
    _, solutions = solve_knapsack(instance)

    # extract meta solutions
    meta_solutions_list = []
    for sc in N:
        meta_solution = []
        for fs in FS:
            weight_feature = 0
            for item in grp[fs]:
                weight_feature += w[sc][item] * solutions[sc][item]
            meta_solution.append(weight_feature)
        meta_solutions_list.append(meta_solution)

    # scale to the size of the knapsack
    meta_solutions_scaled_list = []
    for meta_solution in meta_solutions_list:
        meta_solution_scaled = []
        for entry in meta_solution:
            # we assume here, that the budget is the same for all scenarios
            meta_solution_scaled.append(entry / sum(meta_solution) * budgets[0])
        meta_solutions_scaled_list.append(meta_solution_scaled)

    return meta_solutions_scaled_list


def solve_lr_knapsack_heu_lasso(instance, alpha, sparsity_param=2, global_k=0, column_k=0):
    # use the Lasso heuristic to find a linear surrogate
    if column_k >= 0.5 and global_k >= 0.5:
        raise NotImplementedError
    feat_matrix, budgets, FI, FS, grp, n, N, p, w = instance.get_params()

    # we calculate optimal b values for each instance
    b_true = pp_heu(instance)

    # train a lasso model tp predict b
    if column_k >= 0.5:
        lasso = MultiTaskLasso(alpha=alpha)
    else:
        lasso = Lasso(alpha=alpha)
    scaler = MinMaxScaler()
    scaler.fit(feat_matrix)
    feat_matrix_scaled = scaler.transform(feat_matrix)
    lasso.fit(feat_matrix_scaled, b_true)
    beta = lasso.coef_
    bias = lasso.intercept_

    beta_sparse = np.zeros(beta.shape)

    if column_k >= 0.5:
        sum_cols = []
        for fi in FI:
            sum_cols.append(np.abs(beta[:,fi]).sum())
        arr_sum_cols = np.array(sum_cols)
        idx_sorted = np.argsort(arr_sum_cols)
        idx_largest_k = idx_sorted[-sparsity_param:]
        for fs in FS:
            for fi in idx_largest_k:
                beta_sparse[fs, fi] = beta[fs, fi]

    elif global_k >= 0.5:
        idx_sorted = np.dstack(np.unravel_index(np.argsort(np.abs(beta).ravel()), (len(FS), len(FI))))
        idx_largest_k = idx_sorted[0,-sparsity_param:]
        for (fs, fi) in idx_largest_k:
            beta_sparse[fs, fi] = beta[fs, fi]

    else:
        for fs in FS:
            idx_sorted_row = np.argsort(np.abs(beta[fs,:]))
            idx_largest_k_row = idx_sorted_row[-sparsity_param:]
            for fi in idx_largest_k_row:
                beta_sparse[fs, fi] = beta[fs, fi]

    predictor = LRPredictor(beta_sparse, bias)

    # post-optimize the predictor
    predictor_post = refine_lasso_predictor(instance, predictor)

    return predictor_post


def refine_lasso_predictor(instance, predictor):
    # train a linear predictor only using the same parameters as the given predictor
    b_true = pp_heu(instance)
    b_true_arr = np.array(b_true)

    no_sc = len(instance.N)
    no_fs, no_if = predictor.beta.shape
    lr_models_coeff, lr_models_biases = [], []

    for fs_idx in range(no_fs):
        ifs_mod = np.zeros(instance.ifs.shape)
        for if_idx in range(no_if):
            if predictor.beta[fs_idx, if_idx] > 0.00001 or predictor.beta[fs_idx, if_idx] < -0.00001:
                for sc_idx in range(no_sc):
                    ifs_mod[sc_idx, if_idx] = instance.ifs[sc_idx, if_idx]

        lr_model = LinearRegression().fit(ifs_mod, b_true_arr[:, fs_idx])
        lr_models_coeff.append(lr_model.coef_)
        lr_models_biases.append(lr_model.intercept_)

    beta_new = np.array(lr_models_coeff)
    bias_new = np.array(lr_models_biases)
    new_predictor = LRPredictor(beta_new, bias_new)

    return new_predictor


def evaluate_lr(instance, predictor):
    # evaluate a linear surrogate using the given instance
    feat_matrix, budgets, FI, FS, grp, n, N, p, w = instance.get_params()

    b = []
    for i in N:
        b.append(np.dot(predictor.beta, feat_matrix[i]) + predictor.bias)
    b = np.vstack(b)

    # scaling
    b[b < 0] = 0
    b = b / np.where(b.sum(axis=1, keepdims=True) == 0, 1, b.sum(axis=1, keepdims=True)) * budgets[0]

    m = Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = 1

    x = m.addVars(N, n, vtype=GRB.BINARY, name="x")

    m.setObjective(quicksum(p[i, j] * x[i, j] for i in N for j in n), GRB.MAXIMIZE)

    m.addConstrs(quicksum(x[i, j] * w[i, j] for j in n) <= budgets[i] for i in N)
    m.addConstrs(quicksum(x[i, j] * w[i, j] for j in grp[fs]) <= b[i, fs] for i in N for fs in FS)

    m.optimize()

    return m.ObjVal


def solve_knapsack(instance):
    # solve each scenario as independent knapsack problem and cumulate the objective values
    feat_matrix, budgets, FI, FS, grp, n, N, p, w = instance.get_params()

    objectives = 0
    solutions = np.empty((len(N), len(n)))

    for i in N:
        m = Model("knapsack_opt")
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        x = m.addVars(n, vtype=GRB.BINARY)
        m.setObjective(quicksum(p[i][j] * x[j] for j in n), GRB.MAXIMIZE)
        m.addConstr(quicksum(x[j] * w[i][j] for j in n) <= budgets[i], name="budget")
        m.optimize()
        objectives += m.ObjVal
        for var in m.getVars():
            solutions[i][var.index] = var.X
        m.dispose()
    return objectives, solutions


def solve_knapsack_w_features(p, w, grouping, meta_solutions):
    # solve a knapsack problem for a given instance and meta solution
    no_fs = len(grouping)
    no_items = len(p)
    m = Model("opt_knapsack_features")
    m.Params.OutputFlag = 0
    x = m.addVars(no_items, vtype=GRB.BINARY)
    m.setObjective(quicksum(p[j] * x[j] for j in range(no_items)), GRB.MAXIMIZE)
    m.addConstrs(quicksum(x[j] * w[j] for j in grouping[fs]) <= meta_solutions[fs] for fs in range(no_fs))
    m.optimize()
    return m.ObjVal


def evaluate_dt(instance, dt, meta_solutions):
    # calculate the objective value for a given instance and a decision tree
    feat_matrix, budgets, FI, FS, grp, n, N, p, w = instance.get_params()
    predicted_labels = dt.predict(feat_matrix)

    objective_cum = 0
    for sc in N:
        objective = solve_knapsack_w_features(p[sc], w[sc], grp, meta_solutions[predicted_labels[sc]])
        objective_cum += objective

    return objective_cum



def pp_cart(instance):
    # pre-processing for the CART-based heuristic
    feat_matrix, budgets, FI, FS, grp, n, N, p, w = instance.get_params()

    # solve knapsack problem for each scenario
    _, solutions = solve_knapsack(instance)

    # extract meta solutions
    meta_solutions_list = []
    for sc in N:
        meta_solution = []
        for fs in FS:
            weight_feature = 0
            for item in grp[fs]:
                weight_feature += w[sc][item] * solutions[sc][item]
            meta_solution.append(weight_feature)
        meta_solutions_list.append(meta_solution)

    # scale to the size of the knapsack
    meta_solutions_scaled_list = []
    for meta_solution in meta_solutions_list:
        meta_solution_scaled = []
        for entry in meta_solution:
            # we assume that the budget is always 50!
            meta_solution_scaled.append(entry / sum(meta_solution) * budgets[0])
        meta_solutions_scaled_list.append(meta_solution_scaled)

    # get rid of duplicates
    meta_solutions_scaled_set = set(map(tuple, meta_solutions_scaled_list))
    meta_solutions_cleaned_list = list(map(list, meta_solutions_scaled_set))

    # calculate objective for every combination of scenario and meta solution
    objective_values_matrix = np.zeros((len(N), len(meta_solutions_cleaned_list)))
    for sc in N:
        for idx, meta_solution in enumerate(meta_solutions_cleaned_list):
            objective_value = solve_knapsack_w_features(p[sc], w[sc], grp, meta_solutions_cleaned_list[idx])
            objective_values_matrix[sc][idx] = objective_value

    # calculate a matrix which indicates the difference between best solution and meta solution for every scenario
    difference_matrix = np.zeros(objective_values_matrix.shape)
    for sc in N:
        best_objective = objective_values_matrix[sc].max()
        for meta_solution in range(len(difference_matrix[sc])):
            difference_matrix[sc][meta_solution] = best_objective - objective_values_matrix[sc][meta_solution]

    return objective_values_matrix, meta_solutions_cleaned_list


def solve_dt_knapsack(instance):
    # determine a decision tree which maps scenarios to meta solutions
    feat_matrix, budgets, FI, FS, grp, n, N, p, w = instance.get_params()

    obj_vals, solutions_cleaned = pp_cart(instance)

    # if necessary, reduce number of meta solution considered by the algorithm
    labels = []
    if len(solutions_cleaned) > 4:
        mod = Model(name="reduce_sols_ks")
        mod.Params.OutputFlag = 0
        mod.Params.Threads = 1
        mod.setParam('TimeLimit', 300)
        sols = range(len(solutions_cleaned))
        x = mod.addVars(N, sols, vtype=GRB.INTEGER, name="x")
        y = mod.addVars(sols, vtype=GRB.INTEGER, name="y")
        mod.setObjective(quicksum(obj_vals[sc][so] * x[sc, so] for sc in N for so in sols), GRB.MAXIMIZE)
        mod.addConstrs((quicksum(x[sc, so] for so in sols) == 1 for sc in N), name="assign")
        mod.addConstrs((x[sc, so] - y[so] <= 0 for so in sols for sc in N), name="activate")
        mod.addConstr(quicksum(y[so] for so in sols) <= 4, name="no_sols")
        mod.optimize()

        # get the best solution for each scenario
        # if there are no 4 sols which ensure feasibility for all scenarios use all sols prev. found
        if mod.ObjVal <= 0:
            for scenario in N:
                labels.append(np.argmax(obj_vals[scenario]))
        else:
            for (i, j), v in x.items():
                if v.X > 0.5:
                    labels.append(j)

    else:
        # get the best solution for each scenario
        for sc in N:
            labels.append(np.argmax(obj_vals[sc]))

    dt = DecisionTreeClassifier(max_depth=2)
    dt.fit(feat_matrix, np.array(labels))
    objective_tree = evaluate_dt(instance, dt, solutions_cleaned)

    return objective_tree, dt, solutions_cleaned


# +++++ BENCHMARK +++++


def solve_single_instances(instance):
    # solve a single Knapsack instance
    feat_matrix, budgets, FI, FS, grp, n, N, p, w = instance.get_params()
    objective_all = 0.0

    for i in N:
        m = Model()
        m.Params.OutputFlag = 0
        m.Params.Threads = 1

        x = m.addVars(n, vtype=GRB.BINARY, name="x")

        m.setObjective(quicksum(p[i, j] * x[j] for j in n), GRB.MAXIMIZE)

        m.addConstr(quicksum(x[j] * w[i, j] for j in n) <= budgets[i])

        m.optimize()

        objective_all += m.ObjVal

        m.dispose()

    return objective_all


def solve_one_meta_knapsack(instance, time_limit):
    # solve the problem of finding the best meta solution for a set of Knapsack instances
    feat_matrix, budgets, FI, FS, grp, n, N, p, w = instance.get_params()

    m = Model()
    m.Params.timeLimit = time_limit
    m.Params.OutputFlag = 0
    m.Params.Threads = 1

    b = m.addVars(FS, vtype=GRB.CONTINUOUS, name="b")
    x = m.addVars(N, n, vtype=GRB.BINARY, name="x")

    m.setObjective(quicksum(p[i, j] * x[i, j] for i in N for j in n), GRB.MAXIMIZE)

    m.addConstr(quicksum(b[fs] for fs in FS) <= budgets[0])
    m.addConstrs(quicksum(x[i, j] * w[i, j] for j in grp[fs]) <= b[fs] for i in N for fs in FS)

    m.optimize()

    meta_solution = []
    for fs in FS:
        meta_solution.append(b[fs].X)

    return m.ObjVal, m.MIPGap, meta_solution


def evaluate_one_meta(instance, meta_solution):
    # evaluate a meta solution on an instance
    feat_matrix, budgets, FI, FS, grp, n, N, p, w = instance.get_params()

    # scaling
    meta_solution_scaled = []
    for entry in meta_solution:
        meta_solution_scaled.append(entry / sum(meta_solution) * budgets[0])

    m = Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = 1

    x = m.addVars(N, n, vtype=GRB.BINARY, name="x")

    m.setObjective(quicksum(p[i, j] * x[i, j] for i in N for j in n), GRB.MAXIMIZE)

    m.addConstrs(quicksum(x[i, j] * w[i, j] for j in grp[fs]) <= meta_solution_scaled[fs] for i in N for fs in FS)

    m.optimize()

    return m.ObjVal


def execute_exact(no_train, no_test, no_items, no_fs, seed, sparsity, time_limit, if_parameter, global_k, column_k):
    # execute the exact approach and benchmark methods and store results as json file
    if global_k >= 0.5 and column_k >= 0.5:
        raise NotImplementedError
    # make instance
    tr, te = make_instance(no_train, no_test, no_items, no_fs, seed, if_parameter=if_parameter)
    # do exact lr
    time_start_lr_ex = time.time()
    obj_tr_lr_ex, gap_lr_ex, lr_predictor_ex = solve_lr_knapsack(tr, time_limit, sparsity_param=sparsity,
                                                                 global_k=global_k, column_k=column_k)
    time_lr_ex = time.time() - time_start_lr_ex
    obj_tr_lr_eval_ex = evaluate_lr(tr, lr_predictor_ex)
    obj_te_lr_ex = evaluate_lr(te, lr_predictor_ex)
    # do dt
    time_start_dt = time.time()
    obj_tr_dt, dt_predictor, dt_sols = solve_dt_knapsack(tr)
    time_dt = time.time() - time_start_dt
    obj_te_dt = evaluate_dt(te, dt_predictor, dt_sols)
    # one meta as benchmark
    time_start_one_meta = time.time()
    obj_tr_one_meta, gap_one_meta, one_meta_predictor = solve_one_meta_knapsack(tr, time_limit)
    time_one_meta = time.time() - time_start_one_meta
    obj_te_one_meta = evaluate_one_meta(te, one_meta_predictor)
    # solve everything to optimality
    obj_single_opt_tr = solve_single_instances(tr)
    obj_single_opt_te = solve_single_instances(te)
    # save results
    res_dict = {"no_train": no_train, "no_test": no_test, "no_items": no_items, "no_fs": no_fs, "seed": seed,
                "sparsity": sparsity, "if_parameter": if_parameter, "global_k": global_k, "column_k": column_k,
                "obj_tr_lr_ex": obj_tr_lr_ex, "gap_lr_ex": gap_lr_ex, "obj_tr_lr_eval_ex": obj_tr_lr_eval_ex,
                "time_lr_ex": time_lr_ex, "obj_te_lr_ex": obj_te_lr_ex,
                "obj_tr_dt": obj_tr_dt, "obj_te_dt": obj_te_dt, "time_dt": time_dt,
                "obj_tr_one_meta": obj_tr_one_meta, "gap_one_meta": gap_one_meta, "time_one_meta": time_one_meta,
                "obj_te_one_meta": obj_te_one_meta,
                "obj_single_opt_tr": obj_single_opt_tr, "obj_single_opt_te": obj_single_opt_te,
                "coefficients_lr_ex": lr_predictor_ex.beta.tolist(), "bias_lr_ex": lr_predictor_ex.bias.tolist(),
                "instance_feats_dict": tr.ifs_dict}
    str_file = str(str(no_train) + str(no_test) + str(no_items) + str(no_fs) + str(seed) + str(sparsity)
                   + str(if_parameter) + str(global_k) + str(column_k))
    with open("r_ex_" + str_file + "_x.json", "w") as file:
        json.dump(res_dict, file)


def execute_lasso(no_train, no_test, no_items, no_fs, seed, sparsity, if_parameter, global_k, column_k):
    # execute the Lasso-based heuristic and store results as json file
    if global_k >= 0.5 and column_k >= 0.5:
        raise NotImplementedError
    # values for alpha found by numerical experiments, see file "finding_alpha.py"
    dict_alpha = {('col', 1.0): 2.0, ('col', 2.0): 2.0, ('col', 4.0): 0.5, ('col', 6.0): 0.5, ('glob', 4.0): 1.0,
                  ('glob', 8.0): 0.5, ('glob', 16.0): 0.5, ('glob', 24.0): 0.5, ('row', 1.0): 0.5, ('row', 2.0): 0.5,
                  ('row', 4.0): 0.5, ('row', 6.0): 0.5}
    if global_k >= 0.5:
        alpha = dict_alpha[("glob", sparsity)]
    elif column_k > 0.5:
        alpha = dict_alpha[("col", sparsity)]
    else:
        alpha = dict_alpha[("row", sparsity)]
    # make instance
    tr, te = make_instance(no_train, no_test, no_items, no_fs, seed, if_parameter=if_parameter)
    # do heuristic lasso
    time_start_lr_lasso = time.time()
    lr_predictor_lasso = solve_lr_knapsack_heu_lasso(tr, alpha, sparsity_param=sparsity, global_k=global_k, column_k=column_k)
    time_lr_lasso = time.time() - time_start_lr_lasso
    obj_tr_lr_lasso = evaluate_lr(tr, lr_predictor_lasso)
    obj_te_lr_lasso = evaluate_lr(te, lr_predictor_lasso)
    # solve everything to optimality
    obj_single_opt_tr = solve_single_instances(tr)
    obj_single_opt_te = solve_single_instances(te)
    # save results
    res_dict = {"no_train": no_train, "no_test": no_test, "no_items": no_items, "no_fs": no_fs, "seed": seed,
                "sparsity": sparsity, "if_parameter": if_parameter, "global_k": global_k, "column_k": column_k,
                "obj_tr_lr_lasso": obj_tr_lr_lasso, "obj_te_lr_lasso": obj_te_lr_lasso, "time_lr_lasso": time_lr_lasso,
                "obj_single_opt_tr": obj_single_opt_tr, "obj_single_opt_te": obj_single_opt_te, "alpha": alpha,
                "coefficients_lr_lasso": lr_predictor_lasso.beta.tolist(), "bias_lr_lassp": lr_predictor_lasso.bias.tolist(),
                "instance_feats_dict": tr.ifs_dict}

    str_file = str(str(no_train) + str(no_test) + str(no_items) + str(no_fs) + "_" + str(seed) + "_" + str(sparsity)
                   + str(if_parameter) + str(global_k) + str(column_k))
    with open("r_heu_re_" + str_file + "_x.json", "w") as file:
        json.dump(res_dict, file)

