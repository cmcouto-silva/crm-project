import numpy as np
import pandas as pd
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

def bootstrap_chi2(
    df: pd.DataFrame,
    cat_col1: str,
    cat_col2: str,
    expected: bool = False,
    n_bootstrap: int = 1_000,
    ci: float = .95
    ):
    """
    Estimates confidence intervals for observed or expected frequencies from a contingency
    table using bootstrap resampling.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with categorical data.
    cat_col1 : str
        The first categorical column name.
    cat_col2 : str
        The second categorical column name.
    expected : bool, optional
        If True, returns expected frequencies; if False, returns observed frequencies.
        Default is False.
    n_bootstrap : int, optional
        Number of bootstrap samples to generate. Default is 1000.
    ci : float, optional
        Confidence interval level, between 0 and 1. Default is 0.95.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - First element is a numpy array of lower and upper bounds of confidence intervals
          for the statistics (observed or expected frequencies).
        - Second element is a numpy array of lower and upper bounds of confidence intervals
          for the proportions of these statistics.

    Notes
    -----
    - Performs bootstrap resampling to estimate confidence intervals.
    - For `expected=True`, uses `stats.chi2_contingency` to compute expected frequencies.
    """
    ci_lower = np.round((1-ci)/2, 3)
    ci_upper = 1 - ci_lower

    bootstrap_values = []
    for _ in range(n_bootstrap):
        df_sample = df.sample(frac=1, replace=True)
        df_sample_crosstab = pd.crosstab(df_sample[cat_col1], df_sample[cat_col2])
        if expected:
            bootstrap_values.append(stats.chi2_contingency(df_sample_crosstab)[-1])
        else:
            bootstrap_values.append(df_sample_crosstab)

    bootstrap_values_pct = [bootstrap_sample / bootstrap_sample.sum() for bootstrap_sample in bootstrap_values]

    ci_array = np.quantile(a=np.array(bootstrap_values), q=[ci_lower,ci_upper], axis=0).round().astype(int)
    ci_array_pct = np.quantile(a=np.array(bootstrap_values_pct), q=[ci_lower,ci_upper], axis=0).round(2)

    return ci_array, ci_array_pct


def make_heatmap_freqlabel(
    df: pd.DataFrame,
    index: str,
    columns: str,
    expected: bool = False,
    bootstrap: bool = False,
    **bootstrap_kwargs) -> pd.DataFrame:
    """
    Creates a DataFrame of annotations with frequencies (and optionally bootstrapped confidence intervals)
    for heatmap display, based on observed or expected data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    index : str
        Column name to use as the rows in the crosstab.
    columns : str
        Column name to use as the columns in the crosstab.
    expected : bool, optional
        If True, calculates expected frequencies using chi-squared contingency. Default is False.
    bootstrap : bool, optional
        If True, calculates bootstrapped confidence intervals for the frequencies. Default is False.
    **bootstrap_kwargs
        Additional keyword arguments for the `bootstrap_chi2` function if bootstrapping.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing string values for annotations, formatted to include both absolute
        and percentage frequencies (and confidence intervals if bootstrap is True), suitable for
        annotating a heatmap.

    Notes
    -----
    This function is intended for generating annotated labels for heatmaps, offering detailed
    insights into the observed or expected frequencies of the data. When `bootstrap` is True, it
    enhances the annotations with confidence intervals, providing a deeper statistical understanding.
    """
    df_crosstab = pd.crosstab(df[index], df[columns])

    if expected:
        df_crosstab = (
            pd.DataFrame(stats.chi2_contingency(df_crosstab)[-1], index=df_crosstab.index, columns=df_crosstab.columns)
            .round()
            .astype(int)
        )

    df_crosstab_str = df_crosstab.astype(str)
    df_crosstab_str_pct = (df_crosstab / df_crosstab.sum().sum()*100).round().astype(int).astype(str)+'%'

    if bootstrap:
        ci = bootstrap_chi2(df, index, columns, expected, **bootstrap_kwargs)
        df_crosstab_str = df_crosstab_str + '\n(' + ci[0][0].astype(str) + ' - ' + ci[0][1].astype(str) + ')'

        ci_lower_pct = (ci[1][0]*100).astype(int).astype(str)
        ci_upper_pct = (ci[1][1]*100).astype(int).astype(str)
        df_crosstab_str_pct = df_crosstab_str_pct + '\n(' + ci_lower_pct + '% - ' + ci_upper_pct + '%)'

    return df_crosstab_str + '\n\n' + df_crosstab_str_pct


def plot_chi2_heatmap(
    df: pd.DataFrame,
    index: str,
    columns: str,
    bootstrap: bool = False,
    figsize=(12,8),
    **bootstrap_kwargs
    ):
    """
    Plots heatmaps for observed and expected frequencies of categorical variables in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    index : str
        Column name to use as the rows in the crosstab.
    columns : str
        Column name to use as the columns in the crosstab.
    bootstrap : bool, optional
        If True, includes bootstrapped confidence intervals in the heatmap annotations. Default is False.

    Returns
    -------
    None
        Displays heatmaps for observed and expected frequencies.

    Notes
    -----
    This function generates two heatmaps side by side: one for the observed frequencies and one for
    the expected frequencies under the hypothesis of independence between the two categorical variables.
    It optionally adds bootstrapped confidence intervals to the heatmap annotations if `bootstrap` is True.
    """
    df_absfreq = pd.crosstab(df[index], df[columns])
    df_absfreq_expected = pd.DataFrame(stats.chi2_contingency(df_absfreq)[-1], index=df_absfreq.index, columns=df_absfreq.columns).round().astype(int)

    plot_title = f'{index} vs {columns}'
    fig, ax = plt.subplots(ncols=2, figsize=figsize)

    ax1 = sns.heatmap(df_absfreq, cmap='Blues', annot=make_heatmap_freqlabel(df, index, columns, **bootstrap_kwargs), fmt='', cbar=False, ax=ax[0],linewidths=.5)
    ax1.text(x=0.5, y=1.08, s=plot_title, fontsize='x-large', weight='bold', ha='center', va='bottom', transform=ax1.transAxes)
    ax1.text(x=0.5, y=1.04, s='Observed values', fontsize='small', alpha=0.75, ha='center', va='bottom', transform=ax1.transAxes)
    ax1.tick_params(left=False)

    ax2 = sns.heatmap(df_absfreq_expected, cmap='Blues', annot=make_heatmap_freqlabel(df, index, columns, bootstrap=bootstrap, expected=True, **bootstrap_kwargs), cbar=False, fmt='', ax=ax[1], linewidths=.5)
    ax2.text(x=0.5, y=1.08, s=plot_title, fontsize='x-large', weight='bold', ha='center', va='bottom', transform=ax2.transAxes)
    ax2.text(x=0.5, y=1.04, s='Expected values under independent-association hypothesis', fontsize='small', alpha=0.75, ha='center', va='bottom', transform=ax2.transAxes)
    ax2.tick_params(left=False)
    ax2.set_ylabel(None)
    ax2.set_yticks([])

    plt.tight_layout(w_pad=3)
    plt.show()


def plot_chi2(df, index, columns, ax=None, title=None):
    """
    Plot a histogram of the theoretical chi-square distribution and mark the observed chi-square value.

    Parameters:
        df (pandas.DataFrame): The input dataframe.
        index (str): The column name for the index variable.
        columns (str): The column name for the columns variable.
        ax (matplotlib.axes.Axes, optional): The matplotlib axes to plot on. If not provided, a new figure and axes will be created.
        title (str, optional): The title of the plot. If not provided, a default title will be used.

    Returns:
        matplotlib.axes.Axes: The matplotlib axes object containing the plot.

    """
    # Get observed chi2 and dof
    chi2, p, dof, expected = stats.chi2_contingency(pd.crosstab(df[index], df[columns]))
    # Get theoretical chi2 distribution
    theorical_chi2_dist = np.random.chisquare(dof, df.shape[0])

    # Plot
    sns.histplot(theorical_chi2_dist, kde=True, color='orange', alpha=.3, stat='density', ax=ax)
    ax.axvline(chi2, ls='--', color='black', ymax=.9, label=f'$\chi^2$ = {chi2:.2f}')
    plt.legend(loc='upper right', prop={'size':'small'}, frameon=False)
    
    if title:
        ax.set_title(title, y=1.15, weight='bold', size='large')
    else:
        ax.set_title(f'{index} vs {columns}', y=1.15, weight='bold', size='large')
        
    ax.set_xlabel('$\chi^2$ values', labelpad=10)
    ax.annotate(f'P(X > {chi2:.2f}) = {1-stats.chi2.cdf(chi2, dof):.4f}', xy=(.5,1.06), xycoords='axes fraction', ha='center')
    
    sns.despine(offset=10, trim=True, ax=ax)
    return ax