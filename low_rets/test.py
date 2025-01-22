import statsmodels as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy.stats import norm


def newey_west_se(y, x=None, lags=0):
    """
    Estimate standard errors using Newey-West adjustment for autocorrelation.

    If x is None, we treat it as a test of mean(y)=0 (i.e., alpha>0).
    If x is not None, we treat it as a regression of y on x (like CAPM alpha).
    lags: how many lags in NW adjustment.
          Often ~ floor(4*(n/100)^(2/9)) or some chosen function of sample size.
    Returns:
      (coef, std_err)
    """
    # If no regressor, just test the mean:
    if x is None:
        # Intercept only
        X = np.ones(len(y))
    else:
        # Add intercept
        X = sm.add_constant(x, prepend=True)

    # Fit linear model
    model = OLS(y, X, missing="drop")
    results = model.fit()

    # NW adjusted covariance matrix
    # can set lags automatically, or user-provided
    cov_nw = cov_hac(results, nlags=lags)
    se_nw = np.sqrt(np.diag(cov_nw))

    # The first element is the intercept's coefficient & SE if we have x
    return results.params, se_nw


def rolling_alpha_pvals(returns, window=126, alpha=0.05, lags=0):
    """
    Rolling alpha test (mean > 0) with Newey-West adjustment.
    Returns a Series of p-values for H0: alpha <= 0 vs H1: alpha > 0.

    - returns: pd.Series of daily (or periodic) returns
    - window : rolling window size
    - alpha  : significance level (for reference, though we just return p-vals)
    """
    pvals = []
    idx = returns.index

    for i in range(len(returns)):
        if i < window:
            pvals.append(np.nan)
        else:
            window_data = returns.iloc[i - window : i]
            # We want to test mean(window_data) > 0 with NW
            y = window_data.values
            params, se = newey_west_se(y, x=None, lags=lags)
            # param[0] is the mean, se[0] is NW std error for that mean
            t_stat = params[0] / se[0]
            # One-sided p-value for t_stat>0: p = 1 - Phi(t_stat)
            # But the distribution of OLS param is approx normal for large n
            from scipy.stats import norm

            pval = 1 - norm.cdf(t_stat)
            pvals.append(pval)

    return pd.Series(pvals, index=idx, name="NeweyWest_pval")


def rolling_alpha_pvals_bootstrap(returns, window=126, alpha=0.05, n_bootstrap=100):
    pvals = []
    idx = returns.index

    for i in range(len(returns)):
        if i < window:
            pvals.append(np.nan)
        else:
            window_data = returns.iloc[i - window : i]
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = resample(window_data, replace=True)
                bootstrap_means.append(sample.mean())
            bootstrap_means = np.array(bootstrap_means)
            # Compute p-value
            pval = np.mean(bootstrap_means <= 0)
            pvals.append(pval)

    return pd.Series(pvals, index=idx, name="Bootstrap_pval")


def rolling_alpha_pvals_block_bootstrap(
    returns, window=126, alpha=0.05, n_bootstrap=100, block_size=10
):
    pvals = []
    idx = returns.index

    for i in range(len(returns)):
        if i < window:
            pvals.append(np.nan)
        else:
            window_data = returns.iloc[i - window : i].values
            n = len(window_data)
            # create blocks
            num_blocks = int(np.ceil(n / block_size))
            blocks = [
                window_data[j * block_size : (j + 1) * block_size]
                for j in range(num_blocks)
            ]
            # Remove incomplete blocks
            blocks = [block for block in blocks if len(block) == block_size]
            if not blocks:
                pvals.append(np.nan)
                continue
            # Bootstrap resampling of blocks
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sampled_blocks = resample(
                    blocks,
                    replace=True,
                    n_samples=num_blocks,
                )
                bootstrap_sample = np.concatenate(sampled_blocks)[:n]
                bootstrap_means.append(bootstrap_sample.mean())
            bootstrap_means = np.array(bootstrap_means)
            # Compute p-value
            pval = np.mean(bootstrap_means <= 0)
            pvals.append(pval)

    return pd.Series(pvals, index=idx, name="BlockBootstrap_pval")


def rolling_alpha_pvals_bayesian(
    returns, window=126, alpha=0.05, prior_mean=0, prior_variance=10
):
    pvals = []
    idx = returns.index

    for i in range(len(returns)):
        if i < window:
            pvals.append(np.nan)
        else:
            window_data = returns.iloc[i - window : i]
            data_mean = window_data.mean()
            data_var = window_data.var(ddof=1)
            n = window_data.count()
            # Avoid division by zero
            if data_var == 0:
                posterior_variance = prior_variance
                posterior_mean = prior_mean
            else:
                # Posterior variance
                posterior_variance = 1 / (n / data_var + 1 / prior_variance)
                # Posterior mean
                posterior_mean = posterior_variance * (
                    n * data_mean / data_var + prior_mean / prior_variance
                )

            # posterior probability P(mu > 0)
            z = (0 - posterior_mean) / np.sqrt(posterior_variance)
            pval = norm.cdf(z)
            pvals.append(pval)

    return pd.Series(pvals, index=idx, name="Bayesian_pval")


def visualize_results(results, methods, alpha=0.05, window=126):
    plt.figure(figsize=(14, 6))
    colors = {
        "NeweyWest_pval": "blue",
        "Bootstrap_pval": "orange",
        "BlockBootstrap_pval": "green",
        "Bayesian_pval": "purple",
    }

    for method in methods:
        if method in results.columns:
            plt.plot(
                results.index,
                results[method],
                label=method,
                color=colors.get(method, None),
            )

    plt.axhline(y=alpha, color="red", linestyle="--", label=f"Alpha = {alpha}")
    plt.title("Rolling p-values")
    plt.xlabel("Date")
    plt.ylabel("p-value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pvalues_plot.png")
    plt.show()

    rolling_mean = results["returns"].rolling(window=window).mean()
    results["rolling_mean"] = rolling_mean

    for method in methods:
        if method not in results.columns:
            continue

        plt.figure(figsize=(14, 6))
        ax = plt.gca()

        ax.plot(
            results.index,
            results["rolling_mean"],
            label="Rolling Mean Returns",
            color="black",
            alpha=0.6,
        )

        predicted_positive = results[method] <= alpha
        predicted_negative = results[method] > alpha

        ax.scatter(
            results.index[predicted_positive],
            results["rolling_mean"][predicted_positive],
            label=f"{method} Predicts >0",
            color=colors.get(method, None),
            alpha=0.6,
            marker="o",
            s=50,
        )

        ax.scatter(
            results.index[predicted_negative],
            results["rolling_mean"][predicted_negative],
            label=f"{method} Predicts <=0",
            color="red",
            alpha=0.6,
            marker="x",
            s=50,
        )

        ax.axhline(y=0, color="gray", linestyle="--", label="Mean Return = 0")

        TP = ((predicted_positive) & (results["rolling_mean"] > 0)).sum()
        TN = ((predicted_negative) & (results["rolling_mean"] <= 0)).sum()
        FP = ((predicted_positive) & (results["rolling_mean"] <= 0)).sum()
        FN = ((predicted_negative) & (results["rolling_mean"] > 0)).sum()
        Correct = TP + TN

        textstr = "\n".join(
            (
                f"False Positives (FP): {FP}",
                f"False Negatives (FN): {FN}",
                f"Correct Predictions: {Correct}",
            )
        )

        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
        ax.text(
            0.02,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )

        ax.set_title(f"Rolling Mean Returns with {method} Predictions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rolling Mean Return")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        filename = f"{method}_plot.png"
        plt.savefig(filename)
        plt.close()


def evaluate_results(df, methods, alpha=0.05):
    summary = {}
    for method in methods:
        if method in df.columns:
            significant = df[method] <= alpha
            count = significant.sum()
            summary[method] = count
    summary_df = pd.DataFrame.from_dict(
        summary, orient="index", columns=["Significant Periods"]
    )
    return summary_df


def main():
    csv_file_path = "2021-01-01_1min_mr__returns.csv"

    df = pd.read_csv(csv_file_path, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)

    returns = df["returns"]
    window_size = 126
    alpha_level = 0.05
    n_bootstrap = window_size
    block_size = 10
    prior_mean = 0
    prior_variance = 1

    available_methods = {
        "NeweyWest_pval": rolling_alpha_pvals,
        "Bootstrap_pval": rolling_alpha_pvals_bootstrap,
        "BlockBootstrap_pval": rolling_alpha_pvals_block_bootstrap,
        "Bayesian_pval": rolling_alpha_pvals_bayesian,
    }

    selected_methods = [
        "NeweyWest_pval",
        "Bootstrap_pval",
        "BlockBootstrap_pval",
        "Bayesian_pval",
    ]

    results = pd.DataFrame(index=df.index)
    results["returns"] = returns

    for method in selected_methods:
        print(f"Applying method: {method}")
        if method == "NeweyWest_pval":
            #  lags based on window size
            lags = int(np.floor(4 * (window_size / 100) ** (2 / 9)))
            pvals = available_methods[method](
                returns, window=window_size, alpha=alpha_level, lags=lags
            )
        elif method == "Bootstrap_pval":
            pvals = available_methods[method](
                returns,
                window=window_size,
                alpha=alpha_level,
                n_bootstrap=n_bootstrap,
            )
        elif method == "BlockBootstrap_pval":
            pvals = available_methods[method](
                returns,
                window=window_size,
                alpha=alpha_level,
                n_bootstrap=n_bootstrap,
                block_size=block_size,
            )
        elif method == "Bayesian_pval":
            pvals = available_methods[method](
                returns,
                window=window_size,
                alpha=alpha_level,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
            )
        results = results.join(pvals)

    visualize_results(results, selected_methods, alpha=alpha_level, window=window_size)

    summary = evaluate_results(results, selected_methods, alpha=alpha_level)
    print(summary)

    output_file = "analyzed_returns_comparison.csv"
    results.to_csv(output_file)
    print(f"Results saved to '{output_file}'.")

    for method in selected_methods:
        if method in results.columns:
            significant_periods = results[results[method] <= alpha_level]
            print(f"\nMethod: {method}")
            print(
                f"Number of significant periods: {significant_periods[method].count()}"
            )


if __name__ == "__main__":
    main()
