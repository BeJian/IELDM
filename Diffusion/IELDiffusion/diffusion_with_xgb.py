import numpy as np
from .utils.diffusion import VPSDE, get_pc_sampler
import copy
import xgboost as xgb
from functools import partial
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from .utils.utils_diffusion import build_data_xt, euler_solve, IterForDMatrix, get_xt
from joblib import delayed, Parallel
from tqdm import tqdm


# Class for the diffusion model
class XELDiffusionModel():
    def __init__(self,
                 X,
                 X_covs=None,
                 label_y=None,
                 n_t=50,
                 model='xgb',
                 diffusion_type='vp',  # vp-sde
                 max_depth=7, n_estimators=100, eta=0.3,
                 tree_method='hist', reg_alpha=0.0, reg_lambda=0.0, subsample=1.0,
                 num_leaves=31,
                 duplicate_K=100,
                 bin_indexes=[],
                 cat_indexes=[],
                 int_indexes=[],
                 remove_miss=False,
                 p_in_one=False,
                 true_min_max_values=None,
                 gpu_hist=True,
                 n_z=10,
                 eps=1e-3,
                 beta_min=0.1,
                 beta_max=8,
                 n_jobs=-1,
                 n_batch=1,
                 seed=666,
                 **xgboost_kwargs):

        assert isinstance(X, np.ndarray), "Input dataset must be a Numpy array"
        assert len(X.shape) == 2, "Input dataset must have two dimensions [n,p]"
        assert diffusion_type == 'vp' or diffusion_type == 'flow'
        if X_covs is not None:
            assert X_covs.shape[0] == X.shape[0]
        np.random.seed(seed)

        # Sanity check, must remove observations with only missing data
        obs_to_remove = np.isnan(X).all(axis=1)
        X = X[~obs_to_remove]
        if label_y is not None:
            label_y = label_y[~obs_to_remove]

        # Remove all missing values
        obs_to_remove = np.isnan(X).any(axis=1)
        if remove_miss or (obs_to_remove.sum() == 0):
            X = X[~obs_to_remove]
            if label_y is not None:
                label_y = label_y[~obs_to_remove]
            self.p_in_one = p_in_one
        else:
            self.p_in_one = False

        int_indexes = int_indexes + bin_indexes

        if true_min_max_values is not None:
            self.X_min = true_min_max_values[0]
            self.X_max = true_min_max_values[1]
        else:
            self.X_min = np.nanmin(X, axis=0, keepdims=1)
            self.X_max = np.nanmax(X, axis=0, keepdims=1)

        self.cat_indexes = cat_indexes
        self.int_indexes = int_indexes
        if len(self.cat_indexes) > 0:
            X, self.X_names_before, self.X_names_after = self.dummify(X)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        X = self.scaler.fit_transform(X)

        X1 = X
        self.X_covs = X_covs
        self.X1 = copy.deepcopy(X1)
        self.b, self.c = X1.shape
        if X_covs is not None:
            self.c_all = X1.shape[1] + X_covs.shape[1]
        else:
            self.c_all = X1.shape[1]
        self.n_t = n_t
        self.duplicate_K = duplicate_K
        self.model = model
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.num_leaves = num_leaves
        self.eta = eta
        self.gpu_hist = gpu_hist
        self.label_y = label_y
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.subsample = subsample
        self.n_z = n_z
        self.xgboost_kwargs = xgboost_kwargs

        if model == 'rf' and np.sum(np.isnan(X1)) > 0:
            raise copy.Error('The dataset must not contain missing data in order to use model=random_forest')

        self.diffusion_type = diffusion_type
        self.sde = None
        self.eps = eps
        self.beta_min = beta_min
        self.beta_max = beta_max
        if diffusion_type == 'vp':
            self.sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=n_t)

        self.n_batch = n_batch
        if self.n_batch == 0:
            if duplicate_K > 1:
                X1 = np.tile(X1, (duplicate_K, 1))
                if X_covs is not None:
                    X_covs = np.tile(X_covs, (duplicate_K, 1))

            X0 = np.random.normal(size=X1.shape)  # Noise data
            X_train, y_train = build_data_xt(X0, X1, X_covs, n_t=self.n_t, diffusion_type=self.diffusion_type,
                                             eps=self.eps, sde=self.sde)

        if self.label_y is not None:
            assert np.sum(np.isnan(
                self.label_y)) == 0
            self.y_uniques, self.y_probs = np.unique(self.label_y, return_counts=True)
            self.y_probs = self.y_probs / np.sum(self.y_probs)

            self.mask_y = {}
            for i in range(len(self.y_uniques)):
                self.mask_y[self.y_uniques[i]] = np.zeros(self.b, dtype=bool)
                self.mask_y[self.y_uniques[i]][self.label_y == self.y_uniques[i]] = True
                if self.n_batch == 0:
                    self.mask_y[self.y_uniques[i]] = np.tile(self.mask_y[self.y_uniques[i]], (duplicate_K))
        else:  # assuming a single unique label 0
            self.y_probs = np.array([1.0])
            self.y_uniques = np.array([0])
            self.mask_y = {}
            self.mask_y[0] = np.ones(X1.shape[0], dtype=bool)

        if self.n_batch > 0:
            rows_per_batch = self.b // self.n_batch
            batches = [rows_per_batch for i in range(self.n_batch - 1)] + [self.b - rows_per_batch * (self.n_batch - 1)]
            X1_splitted = {}
            X_covs_splitted = {}
            for i in self.y_uniques:
                X1_splitted[i] = np.split(X1[self.mask_y[i], :], batches, axis=0)
                if X_covs is not None:
                    X_covs_splitted[i] = np.split(X_covs[self.mask_y[i], :], batches, axis=0)
                else:
                    X_covs_splitted[i] = None

        n_steps = n_t
        n_y = len(self.y_uniques)
        t_levels = np.linspace(eps, 1, num=n_t)

        if self.p_in_one:
            if self.n_jobs == 1:
                self.regr = [[None for i in range(n_steps)] for j in self.y_uniques]
                for i in tqdm(range(n_steps)):
                    for j in range(len(self.y_uniques)):
                        if self.n_batch > 0:  # Data iterator, no need to duplicate, not make xt yet
                            self.regr[j][i] = self.train_iterator(X1_splitted[j], X_covs_splitted[j], t=t_levels[i],
                                                                  dim=None)
                        else:
                            self.regr[j][i] = self.train_parallel(
                                X_train.reshape(self.n_t, self.b * self.duplicate_K, self.c_all)[i][self.mask_y[j], :],
                                y_train.reshape(self.b * self.duplicate_K, self.c)[self.mask_y[j], :]
                            )
            else:
                if self.n_batch > 0:
                    self.regr = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.train_iterator)(X1_splitted[j], X_covs_splitted[j], t=t_levels[i], dim=None) for i
                        in tqdm(range(n_steps)) for j in self.y_uniques)
                else:
                    self.regr = Parallel(n_jobs=self.n_jobs)(  # using all cpus
                        delayed(self.train_parallel)(
                            X_train.reshape(self.n_t, self.b * self.duplicate_K, self.c_all)[i][self.mask_y[j], :],
                            y_train.reshape(self.b * self.duplicate_K, self.c)[self.mask_y[j], :]
                        ) for i in tqdm(range(n_steps)) for j in self.y_uniques
                    )
                # Replace fits with doubly loops to make things easier
                self.regr_ = [[None for i in range(n_steps)] for j in self.y_uniques]
                current_i = 0
                for i in range(n_steps):
                    for j in range(len(self.y_uniques)):
                        self.regr_[j][i] = self.regr[current_i]
                        current_i += 1
                self.regr = self.regr_
        else:
            if self.n_jobs == 1:
                self.regr = [[[None for k in range(self.c)] for i in range(n_steps)] for j in self.y_uniques]
                for i in tqdm(range(n_steps)):
                    for j in range(len(self.y_uniques)):
                        for k in range(self.c):
                            if self.n_batch > 0:  # Data iterator, no need to duplicate, not make xt yet
                                self.regr[j][i][k] = self.train_iterator(X1_splitted[j], X_covs_splitted[j],
                                                                         t=t_levels[i], dim=k)
                            else:
                                self.regr[j][i][k] = self.train_parallel(
                                    X_train.reshape(self.n_t, self.b * self.duplicate_K, self.c_all)[i][self.mask_y[j],:],
                                    y_train.reshape(self.b * self.duplicate_K, self.c)[self.mask_y[j], k]
                                )
            else:
                if self.n_batch > 0:  # Data iterator, no need to duplicate, not make xt yet
                    self.regr = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.train_iterator)(X1_splitted[j], X_covs_splitted[j], t=t_levels[i], dim=k) for i in
                        tqdm(range(n_steps)) for j in self.y_uniques for k in range(self.c))
                else:
                    self.regr = Parallel(n_jobs=self.n_jobs)(  # using all cpus
                        delayed(self.train_parallel)(
                            X_train.reshape(self.n_t, self.b * self.duplicate_K, self.c_all)[i][self.mask_y[j], :],
                            y_train.reshape(self.b * self.duplicate_K, self.c)[self.mask_y[j], k]
                        ) for i in tqdm(range(n_steps)) for j in self.y_uniques for k in range(self.c)
                    )
                # Replace fits with doubly loops to make things easier
                self.regr_ = [[[None for k in range(self.c)] for i in range(n_steps)] for j in self.y_uniques]
                current_i = 0
                for i in range(n_steps):
                    for j in range(len(self.y_uniques)):
                        for k in range(self.c):
                            self.regr_[j][i][k] = self.regr[current_i]
                            current_i += 1
                self.regr = self.regr_

    def train_iterator(self, X1_splitted, X_covs_splitted, t, dim):
        np.random.seed(self.seed)

        it = IterForDMatrix(X1_splitted, X_covs_splitted, t=t, dim=dim, n_batch=self.n_batch, n_epochs=self.duplicate_K,
                            diffusion_type=self.diffusion_type, eps=self.eps, sde=self.sde)
        data_iterator = xgb.QuantileDMatrix(it)

        xgb_dict = {'objective': 'reg:squarederror', 'eta': self.eta, 'max_depth': self.max_depth,
                    "reg_lambda": self.reg_lambda, 'reg_alpha': self.reg_alpha, "subsample": self.subsample,
                    "seed": self.seed,
                    "tree_method": self.tree_method, 'device': 'cuda' if self.gpu_hist else 'cpu',
                    "device": "cuda" if self.gpu_hist else 'cpu'}
        for myarg in self.xgboost_kwargs:
            xgb_dict[myarg] = self.xgboost_kwargs[myarg]
        out = xgb.train(xgb_dict, data_iterator, num_boost_round=self.n_estimators)

        return out

    def train_parallel(self, X_train, y_train):

        if self.model == 'rf':
            out = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                        random_state=self.seed)
        elif self.model == 'lgbm':
            out = LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves, learning_rate=0.1,
                                random_state=self.seed, force_col_wise=True, verbose=-1)
        elif self.model == 'catb':
            out = CatBoostRegressor(iterations=self.n_estimators, loss_function='RMSE', max_depth=self.max_depth,
                                    silent=True,
                                    l2_leaf_reg=0.0,
                                    random_seed=self.seed)  # consider t as a golden feature if t is a variable
        elif self.model == 'xgb':
            out = xgb.XGBRegressor(n_estimators=self.n_estimators, objective='reg:squarederror', eta=self.eta,
                                   max_depth=self.max_depth,
                                   reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, subsample=self.subsample,
                                   seed=self.seed, tree_method=self.tree_method,
                                   device='cuda' if self.gpu_hist else 'cpu', **self.xgboost_kwargs)
        else:
            raise Exception("model value does not exists")

        if len(y_train.shape) == 1:
            y_no_miss = ~np.isnan(y_train)
            out.fit(X_train[y_no_miss, :], y_train[y_no_miss])
        else:
            out.fit(X_train, y_train)

        return out

    def dummify(self, X):
        df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])  # to Pandas
        df_names_before = df.columns
        for i in self.cat_indexes:
            df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=True)
        df_names_after = df.columns
        df = df.to_numpy()
        return df, df_names_before, df_names_after

    def unscale(self, X):
        if self.scaler is not None:
            X = self.scaler.inverse_transform(X)
        return X

    def clean_onehot_data(self, X):
        if len(self.cat_indexes) > 0:
            X_names_after = copy.deepcopy(self.X_names_after.to_numpy())
            prefixes = [x.split('_')[0] for x in self.X_names_after if '_' in x]
            unique_prefixes = np.unique(prefixes)
            for i in range(len(unique_prefixes)):
                cat_vars_indexes = [unique_prefixes[i] + '_' in my_name for my_name in self.X_names_after]
                cat_vars_indexes = np.where(cat_vars_indexes)[0]
                cat_vars = X[:, cat_vars_indexes]
                cat_vars = np.concatenate((np.ones((cat_vars.shape[0], 1)) * 0.5, cat_vars), axis=1)
                max_index = np.argmax(cat_vars, axis=1)
                X[:, cat_vars_indexes[0]] = max_index
                X_names_after[cat_vars_indexes[0]] = unique_prefixes[i]
            df = pd.DataFrame(X, columns=X_names_after)
            df = df[self.X_names_before]
            X = df.to_numpy()
        return X

    def clip_extremes(self, X):
        if self.int_indexes is not None:
            for i in self.int_indexes:
                X[:, i] = np.round(X[:, i], decimals=0)
        small = (X < self.X_min).astype(float)
        X = small * self.X_min + (1 - small) * X
        big = (X > self.X_max).astype(float)
        X = big * self.X_max + (1 - big) * X
        return X

    def predict_over_c(self, X, i, j, k, dmat, expand=False, X_covs=None):
        if X_covs is not None:
            X = np.concatenate((X, X_covs), axis=1)
        if dmat:
            X_used = xgb.DMatrix(data=X)
        else:
            X_used = X
        if k is None:
            return self.regr[j][i].predict(X_used)
        elif expand:
            return np.expand_dims(self.regr[j][i][k].predict(X_used), axis=1)  # [b, 1]
        else:
            return self.regr[j][i][k].predict(X_used)

    def my_model(self, t, y, mask_y=None, dmat=False, unflatten=True, X_covs=None):
        if unflatten:
            # y is [b*c]
            c = self.c
            b = y.shape[0] // c
            X = y.reshape(b, c)
        else:
            X = y

        out = np.zeros(X.shape)
        i = int(round(t * (self.n_t - 1)))
        for j, label in enumerate(self.y_uniques):
            if X_covs is not None:
                X_covs_masked = X_covs[mask_y[label], :]
            else:
                X_covs_masked = None
            if mask_y[label].sum() > 0:
                if self.p_in_one:
                    out[mask_y[label], :] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=None, dmat=dmat,
                                                                X_covs=X_covs_masked)
                else:
                    for k in range(self.c):
                        out[mask_y[label], k] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=k, dmat=dmat,
                                                                    X_covs=X_covs_masked)

        if self.diffusion_type == 'vp':
            alpha_, sigma_ = self.sde.marginal_prob_coef(X, t)
            out = - out / sigma_
        if unflatten:
            out = out.reshape(-1)
        return out

    def generate(self, batch_size=None, n_t=None, X_covs=None):

        if X_covs is not None:
            assert X_covs.shape[0] == batch_size

        y0 = np.random.normal(size=(self.b if batch_size is None else batch_size, self.c))
        expected_counts = np.round(self.y_probs * y0.shape[0]).astype(int)
        difference = y0.shape[0] - expected_counts.sum()
        if difference != 0:
            max_prob_index = np.argmax(self.y_probs)
            expected_counts[max_prob_index] += difference

        label_y = np.repeat(self.y_uniques, expected_counts)
        mask_y = {}  # mask for which observations has a specific value of y
        for i in range(len(self.y_uniques)):
            mask_y[self.y_uniques[i]] = np.zeros(y0.shape[0], dtype=bool)
            mask_y[self.y_uniques[i]][label_y == self.y_uniques[i]] = True
        my_model = partial(self.my_model, mask_y=mask_y, dmat=self.n_batch > 0, X_covs=X_covs)

        if self.diffusion_type == 'vp':
            sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.n_t if n_t is None else n_t)
            ode_solved = get_pc_sampler(my_model, sde=sde, denoise=True, eps=self.eps)(y0.reshape(-1))
        else:
            ode_solved = euler_solve(my_model=my_model, y0=y0.reshape(-1),
                                     N=self.n_t if n_t is None else n_t)
        solution = ode_solved.reshape(y0.shape[0], self.c)
        solution = self.unscale(solution)
        solution = self.clean_onehot_data(solution)
        solution = self.clip_extremes(solution)

        if self.label_y is not None:
            solution = np.concatenate((solution, np.expand_dims(label_y, axis=1)), axis=1)

        return solution

