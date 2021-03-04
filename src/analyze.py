import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from src.stat_funcs import Statistics


class FactorisationMachine:
    def __init__(self, k=3):
        self.w0 = None
        self.w = None
        self.v = None
        self.k = k
        self.learn_rate = 0.01
        self.statistics = Statistics()
        self.b_size = 100

    def fit(self, X, y):
        self.w0 = 0.5
        self.w = np.full((X.shape[1], 1), 0.5)
        self.v = np.full((X.shape[1], self.k), 0.5)

        self.batch_grad_descent(X, y)
        rmse = self._count_loss_func(X, y)

    def batch_grad_descent(self, X, y):
        batches = self._create_batch(X, y)
        i = 1
        batch_num = X.shape[0] // self.b_size
        bar = tqdm(range(batch_num), total=batch_num)
        for x_batch, y_batch in batches:
            error = y_batch - self._predict(x_batch)
            lin_p = 2 * self.learn_rate

            self.w0 += lin_p * np.sum(error)
            self.w += lin_p * x_batch.T.dot(error)
            self.v += lin_p * (x_batch.T.dot(np.multiply(error, (x_batch.dot(self.v)))) -
                               np.multiply(self.v, x_batch.T.power(2).dot(error)))

            bar.update()
            i += 1
            if np.max(np.abs(error)) < 1e-20:
                print(f"i - {i}, error - {error}")
                break


    def predict_and_analyze(self, X, y):
        y_predict = self._predict(X)
        rmse, r2 = self.statistics.get_statistics(y, y_predict)
        return y_predict, rmse, r2

    def analyze_results(self, res):
        mean = np.mean(res, axis=0).tolist()
        var = np.var(res, axis=0, ddof=0).tolist()
        res.append(mean)
        res.append(var)
        columns = ['r2 test', 'RMSE test', 'r2 train', 'RMSE train']
        results_df = pd.DataFrame(res,
                                  index=['T1', 'T2', 'T3', 'T4', 'T5', 'E', 'STD'],
                                  columns=columns)
        results_df.to_csv("data/results.csv", header=True, index=False)
        print(tabulate(results_df, headers='keys', tablefmt='psql'))

    def _predict(self, X):
        quadratic = 0.5 * np.sum(((X.dot(self.v)) ** 2) - (X.power(2).dot(self.v ** 2)), axis=1)
        return self.w0 + X.dot(self.w) + quadratic.reshape(-1, 1)

    def _count_loss_func(self, X, y):
        rmse = 0
        batch = self._create_batch(X, y)
        batch_num = X.shape[0] // self.b_size
        for x_batch, y_batch in batch:
            rmse += self.statistics.rmse(self._predict(x_batch), y_batch)
        rmse /= batch_num
        return rmse

    def _create_batch(self, X, y):
        batch_num = X.shape[0] // self.b_size
        X_b = (X[i * self.b_size: (i + 1) * self.b_size, :] for i in (range(batch_num)))
        y_b = (y[i * self.b_size: (i + 1) * self.b_size, :] for i in (range(batch_num)))
        return zip(X_b, y_b)

    # def fit(self, block, n_factors=15):
    #     features = block[:, 1:]
    #     target = block[:, 0]
    #     n = block.shape[0]
    #     print(f'{n}  {features.shape[1]}')
    #     weight_k = np.ones(features.shape[1])
    #     v_k = np.ones((n_factors, n_factors))
    #     iteration = 1
    #     Q = 0
    #     while Q < 0.3 and iteration < 10000:
    #         loss = self.count_loss(features, weight_k, v_k, target)
    #         pred = self.predict(features, weight_k, v_k, n_factors, i)
    #         pred, summed = self.predict_instance(features, weight_k, v_k, n_factors, i)
    #
    #         # update weight
    #         grad = np.dot(features.T, loss) / np.sqrt(n * np.sum(loss ** 2))
    #         weight_k = weight_k - grad / iteration
    #
    #         for i in range(features.shape[0]):
    #             # update factor
    #             for factor in range(n_factors):
    #                 term = summed[factor] - v_k[factor, :] * features
    #                 v_k[factor, :] = loss_gradient * data[index] * term
    #
    #         Q = self.count_r_2(target, self.count_loss(features, weight, v_k, target))
    #         iteration += 1
    #     return weight_k
    #
    # def predict(self, X, w, v):
    #     linear_output = w * X
    #     term = (X * v) ** 2 - (X.power(2) * (v ** 2))
    #     factor_output = 0.5 * np.sum(term, axis=1)
    #     return linear_output + factor_output
    #
    # def generate_result(self, block, blocks_to_train, i, weight, v_k):
    #     features_train = blocks_to_train[:, :-1]
    #     target_train = blocks_to_train[:, -1]
    #     # train_loss = self.count_loss(features_train, weight, v_k, target_train)
    #     # rmse_train = self.count_rmse(features_train, train_loss)
    #     # r_2_train = self.count_r_2(target_train, train_loss)
    #     #
    #     # features_test = block[:, :-1]
    #     # target_test = block[:, -1]
    #     # test_loss = self.count_loss(features_test, weight, v_k, target_test)
    #     # r_2_test = self.count_r_2(target_test, test_loss)
    #     # rmse_test = self.count_rmse(features_test, test_loss)
    #
    #     # result = [r_2_test, rmse_test, r_2_train, rmse_train]
    #     # result.extend(weight)
    #     return result
    #
    # # compute result including RMSE and R2 for each cycle of CV
    #
    #
    # def cross_validation(self, dataset_df):
    #     dataset = np.c_[np.ones(dataset_df.shape[0]), dataset_df]
    #     blocks = np.array_split(dataset[:], 5)
    #     results = []
    #     for i in range(5):
    #         block = blocks[i]
    #         blocks_to_train = np.concatenate([x for x in blocks if x is not block])
    #         weight, v = self.factorization_machine(blocks_to_train)
    #
    #         result = self.generate_result(block, blocks_to_train, i, weight, v)
    #         results.append(result)
    #     return results
    #
    # def predict_instance(self, features, w, v, n_factors):
    #     summed = np.zeros(n_factors)
    #     summed_squared = np.zeros(n_factors)
    #
    #     # linear output w * x
    #     pred = features * w
    #
    #     # factor output
    #     for factor in range(n_factors):
    #         term = v[factor, :] * features
    #         summed[factor] += term
    #         summed_squared[factor] += term * term
    #         pred += 0.5 * (summed[factor] * summed[factor] - summed_squared[factor])
    #     return pred, summed
