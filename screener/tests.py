#
import numpy
from scipy.stats import shapiro, skew, kurtosis, kstest
from scipy.special import kl_div
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import normal_ad, het_breuschpagan, het_white, acorr_ljungbox, acorr_breusch_godfrey
from statsmodels.stats.oneway import test_scale_oneway


#


#
def smape_func(y_true, y_pred):
    return (numpy.abs(y_true - y_pred) / ((numpy.abs(y_true) + numpy.abs(y_pred)) / 2)).sum() / y_true.shape[0]


def check_up(y, y_hat, model, x):

    errors = y - y_hat

    # normality

    ad = 1 - normal_ad(x=errors)[1]
    sw = 1 - shapiro(x=errors)[1]
    sk = numpy.abs(numpy.tanh(skew(a=errors)))
    ku = numpy.abs(numpy.tanh(kurtosis(a=errors)))

    # homoscedasticity

    """
    bp_iid = het_breuschpagan(resid=errors,
                              exog_het=numpy.concatenate((x, numpy.ones(shape=(x.shape[0], 1))), axis=1),
                              robust=True)[1]
    bp_nrm = het_breuschpagan(resid=errors,
                              exog_het=numpy.concatenate((x, numpy.ones(shape=(x.shape[0], 1))), axis=1),
                              robust=False)[1]
    wh = het_white(resid=errors,
                   exog=numpy.concatenate((x, numpy.ones(shape=(x.shape[0], 1))), axis=1))[1]
    """

    sampled = [resample(errors, replace=True, n_samples=errors.shape[0]) for _ in range(2)]
    lev_trim = 1 - test_scale_oneway(data=sampled, method='equal', center='trimmed').pvalue
    bf = 1 - test_scale_oneway(data=sampled, method='bf', center='median').pvalue

    # stationarity

    adf = adfuller(x=errors, regression='c', autolag='AIC')[1]

    # independency on Y

    lm = OLS(endog=errors, exog=y).fit()
    ols_sgnf = 1 - lm.f_pvalue
    ols_r = r2_score(y_true=errors, y_pred=lm.predict(exog=y))
    ols_r = ols_r if ols_r >= 0 else 1

    model.fit(X=y.reshape(-1, 1), y=errors.reshape(-1, 1))

    predicted = model.predict(X=y.reshape(-1, 1))
    model_r = r2_score(y_true=errors[(errors.shape[0] - predicted.shape[0]):], y_pred=predicted)
    model_r = model_r if model_r >= 0 else 1

    # independency internal
    # lb_orig = acorr_ljungbox(x=errors, auto_lag=True)
    lb_orig = 1 - acorr_ljungbox(x=errors, lags=40)['lb_pvalue'].min()

    # bg_orig = acorr_breusch_godfrey(res=errors, nlags=40)[1]
    # bg_orig = numpy.nan

    # lb_boost = max([acorr_ljungbox(x=x, auto_lag=True) for x in samples])
    ## lb_boost = 1 - min([acorr_ljungbox(x=resample(errors, replace=True, n_samples=errors.shape[0]),
    ##                           lags=40)['lb_pvalue'].min() for _ in range(1_000)])
    # bg_boost = max([acorr_breusch_godfrey(res=x, nlags=40) for x in samples])
    # bg_boost = numpy.nan

    # distance to Y

    y_quantiles = ECDF(x=y)
    y_hat_quantiles = ECDF(x=y_hat)
    y_quantiles, y_hat_quantiles = [y_quantiles(x) for x in y], [y_hat_quantiles(x) for x in y_hat]
    ks = 1 - kstest(rvs=y_hat, cdf=y, alternative='two-sided', mode='auto')[1]
    kl = 1 - numpy.exp(-kl_div(y_hat_quantiles, y_quantiles).sum())

    # measures

    r2_orig = r2_score(y_true=y, y_pred=y_hat)
    r2_adj = 1 - (1 - r2_orig) * (x.shape[0] - 1) / (x.shape[0] - x.shape[1])
    r2_orig = 1 - r2_orig if r2_orig >= 0 else 1
    r2_adj = 1 - r2_adj if r2_adj >= 0 else 1

    smape = numpy.tanh(smape_func(y_true=y, y_pred=y_hat))

    #

    results = {
        'Anderson-Darling test': ad,
        'Shapiro-Wilk test': sw,
        'Skewness statistic': sk,
        'Kurtosis': ku,
        "Levene's test": lev_trim,
        'Brown–Forsythe test test': bf,
        'Augmented Dickey-Fuller test': adf,
        'OLS regression on Y significance statistic': ols_sgnf,
        'OLS regression on Y R-squared statistic': ols_r,
        'Self-Model regression on Y R-squared statistic': model_r,
        'Ljung-Box autocorrelation test': lb_orig,
        # 'Ljung-Box bootstrapped autocorrelation test': lb_boost,
        'Kolmogorov-Smirnov statistic': ks,
        'Kullback-Leibler statistic': kl,
        'R-squared measure': r2_orig,
        'R-squared adjusted measure': r2_adj,
        'SMAPE measure': smape,
    }

    return results
