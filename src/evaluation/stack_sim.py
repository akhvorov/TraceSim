import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Iterable, Tuple

from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve

from methods.base import SimStackModel


def binom_interval(success: int, total: int, err=0.05):
    quantile = err / 2.
    lower = beta.ppf(quantile, success, total - success + 1)
    upper = beta.ppf(1 - quantile, success + 1, total - success)
    return lower, upper


def bootstrap_bin_metric(y_true: List[int], y_pred: List[float],
                         metric: Callable[[Iterable, Iterable], float],
                         err: float = 0.05, iters: int = 100, size: int = 1):
    values = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    real_value = metric(y_true, y_pred)
    n = len(y_true)
    sn = int(size * n)
    left = int(iters * err / 2)
    while len(values) < iters:
        inds = np.random.choice(n, sn)
        try:
            value = metric(y_true[inds], y_pred[inds])
            values.append(value)
        except:
            pass
    values = sorted(values)
    return round(real_value, 4), round(values[left], 4), round(values[iters - 1 - left], 4)


def metric_in_time(y_tr: List[int], y_pr: List[int],
                   metric: Callable[[Iterable, Iterable], float],
                   buckets: int = 10, with_ci: bool = False):
    n = len(y_pr)
    step = int(n / buckets) + 1
    values = []
    for i in range(0, n, step):
        if with_ci:
            try:
                value = bootstrap_bin_metric(y_tr[i:i + step], y_pr[i:i + step], metric)
            except ValueError as e:
                value = values[-1]
                print("Valerr")
        else:
            value = round(metric(y_tr[i:i + step], y_pr[i:i + step]), 4)
        values.append(value)
    if with_ci:
        return list(zip(*values))
    return values


def draw_auc_at_time(y_tr: List[int], y_pr: List[int]):
    time_auc, lauc, rauc = metric_in_time(y_tr, y_pr, roc_auc_score, with_ci=True)
    ax = plt.subplot(222)
    x = range(len(time_auc))
    ax.plot(x, time_auc, color='b')
    ax.fill_between(x, lauc, rauc, color='b', alpha=.1)
    ax.set_ylim((0, 1))


def auc_model(model: SimStackModel, data: Iterable[Tuple[int, int, int]], full: bool = False, silent: bool = False):
    data = sorted(data, key=lambda s: max(s[0], s[1]))
    y_true = [l for _, _, l in data]
    y_pred = model.predict_pairs(data)

    auc, l, r = bootstrap_bin_metric(y_true, y_pred, roc_auc_score)
    if not silent:
        print(f"{auc}: [{l}, {r}]")

    if full:
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"AUC = {auc}")

        fpr, tpr, th = roc_curve(y_true, y_pred)
        ax = plt.subplot(221)
        ax.plot(fpr, tpr)

        draw_auc_at_time(y_true, y_pred)
        plt.show()
    return auc, l, r
