from scipy.stats import pearsonr
import os
import seaborn as sns
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_select import golden_select, ats_select, zero_shot_select, subtuning_select, model_size_select, our_fit_select, vanilla_fit_select
from fit_law import fit_our_law, fit_vanilla_law, our_law_transform, vanilla_law_transform

PEARSON_METRIC =  'PearCorr'
RELACC_METRIC =  'RelAcc'
OUR_LAW_METHOD = 'OurFit'
ATS_METHOD = 'AtS'
ZERO_METHOD =  'ZeroShot'
SUBTUNING_METHOD =  'SubTuning'
SIZE_METHOD =  'ModelSize'
VANILLA_LAW_METHOD = 'VanillaFit'


def load_benchmark(dataset_name, size_constraint=None, remove_largest=None):
    df = pd.read_csv(f'benchmark/{dataset_name}.csv',
                     index_col=0)
    df.columns = [0,200,400,800,1600,3200,6400,12800,25600,51200,102400,204800,409600,819200,1638400,'size','family']

    # remove model with size constraints
    if size_constraint is not None:
        df = df[df['size'] < size_constraint]
    elif remove_largest is not None:
        df['size_rank'] = df['size'].rank(ascending=False)
        df = df[df['size_rank'] > remove_largest]
        df = df.drop(columns=['size_rank'])

    return df
    


def selection_performance(dataset_lst=['wmt19', 'gigaword', 'flan'], 
                          method_lst=[OUR_LAW_METHOD, ATS_METHOD, ZERO_METHOD, SUBTUNING_METHOD, SIZE_METHOD, VANILLA_LAW_METHOD], 
                          size_constraint=None, remove_largest=None):
    method_dict = {
        OUR_LAW_METHOD: our_fit_select,
        ATS_METHOD: ats_select,
        ZERO_METHOD: zero_shot_select,
        SUBTUNING_METHOD: subtuning_select,
        SIZE_METHOD: model_size_select,
        VANILLA_LAW_METHOD: vanilla_fit_select
    }

    res = []
    predict_data_num = 1638400
    for dataset_name in dataset_lst:
        df = load_benchmark(dataset_name, size_constraint, remove_largest)
        dataset_name = dataset_name.upper()
        for max_data_num in [204800, 102400, 51200, 25600, 12800, 6400, 3200]:
            ratio = int(predict_data_num / max_data_num)

            golden_rank, golden_best, golden_worst, golden_performance = golden_select(df, predict_data_num)
            df['golden_rank'] = df[predict_data_num].rank(ascending=True)

            for method in method_lst:
                rank, feature = method_dict[method](df, max_data_num, predict_data_num)
                best = df.iloc[rank.argmin()][predict_data_num]
                pearson_corr = round(pearsonr(feature, golden_performance)[0], 4)
                relative_acc = round((golden_worst - best) / (golden_worst - golden_best), 4)
                res.append({'dataset': dataset_name, 
                            'ratio': ratio,
                            'max_data_num': max_data_num,
                            'predict_data_num': predict_data_num,
                            'metric': PEARSON_METRIC,
                            'method': method,
                            'value': pearson_corr})
                res.append({'dataset': dataset_name,
                            'ratio': ratio,
                            'max_data_num': max_data_num,
                            'predict_data_num': predict_data_num,
                            'metric': RELACC_METRIC,
                            'method': method,
                            'value': relative_acc})

    return pd.DataFrame(res)

def fig1_a():
    """
    A graph to demonstrate the difficulty of the problem
    """
    sns.set_theme(rc={"legend.fontsize": 12})
    sns.set_theme(style="whitegrid")

    res = selection_performance(method_lst=[ZERO_METHOD, SUBTUNING_METHOD, SIZE_METHOD])
    res = res[res['metric'] == PEARSON_METRIC]
    

    g = sns.catplot(
        data=res, kind="bar",
        x="ratio", y="value", hue="method",
        errorbar="sd", palette="dark", 
        alpha=.6, 
        height=6,
        aspect=2,
        legend_out=False,
    )

    g.axes[0][0].set_ylabel('Pearson Correlation')
    g.despine(left=True)
    g.set_titles("")
    g.set_xlabels("Data Budget Ratio")

    plt.savefig('results/fig1_a.png', dpi=300)


def fig1_b():
    # Second part: the scatter plot of six models with line
    sns.set_theme(style="whitegrid")

    dataset = 'flan'
    df = load_benchmark(dataset)
    df.reset_index()

    # add a column 'dataset'
    df['dataset'] = dataset.upper()
    
    all_df = df
    all_df = all_df.drop(columns=[0]) # drop the zero-shot
    all_df = all_df.reset_index()
    origin_df = all_df.copy(deep=True)
    all_df = all_df.melt(id_vars=['dataset', 'size', 'config name', 'family'], var_name='# Samples', value_name='Test Loss') # melt the dataframe to a long format

    all_df['size'] = np.log(all_df['size']) # log scale
    all_df = all_df.rename(columns={'config name': 'Model'})

    model_lst = [
        "cerebras/Cerebras-GPT-256M",
        "gpt2",
    ]

    all_df = all_df[all_df['Model'].isin(model_lst)]

    g = sns.relplot(
        data=all_df,
        x="# Samples", 
        y="Test Loss",
        hue="Model",
        kind='scatter',
        legend='brief',
        alpha=0.5,
        aspect=2,
        facet_kws={'sharey': False, 'sharex': True},
        s=200
    )

    # Plot laws
    handles, labels = g.axes[0][0].get_legend_handles_labels()

    dataset = dataset.upper()
    cur_ax = g.axes[0][0]
    for model in model_lst:
        cur_color = handles[labels.index(model)].get_color()

        df = origin_df[origin_df['dataset'] == dataset]
        df = df.drop(columns=['size', 'family', 'dataset']) # drop unused columns
        df_model = df[df['config name'] == model].iloc[0]
        df_model = df_model.drop(labels=['config name'])

        x_plot = np.linspace(min(np.log(df_model.index.values.astype(float))), max(np.log(df_model.index.values.astype(float))), 100000)

        # fit our law
        fitted_params, bestloss = fit_our_law(np.log(df_model.index.values.astype(float)), np.log(df_model.to_numpy(dtype=np.float32)))

        # plot our law
        y_plot = our_law_transform((x_plot), *fitted_params)
        cur_ax.plot(np.exp(x_plot), np.exp(y_plot), color=cur_color, linewidth=3, linestyle='-')

        # fit vanilla law
        fitted_params, bestloss = fit_vanilla_law(np.log(df_model.index.values.astype(float)), np.log(df_model.to_numpy(dtype=np.float32)))

        # plot vanilla law
        y_plot = vanilla_law_transform((x_plot), *fitted_params)
        cur_ax.plot(np.exp(x_plot), np.exp(y_plot), color=cur_color, linewidth=3, linestyle='--')

    g.set_titles("{col_name}")

    # set log scale
    g.axes[0][0].set_xscale('log')
    g.axes[0][0].set_yscale('log')

    # set ticks
    g.axes[0][0].set_yticks([2.0, 2.5, 3.0, 3.5, 4.0])
    g.axes[0][0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    g.savefig('results/fig1_b.png', dpi=800)


def fig3():
    sns.set_theme(style="whitegrid")

    all_df = []
    for dataset in ['wmt19', 'gigaword', 'flan']:
        df = load_benchmark(dataset)
        df.reset_index()

        # add a column 'dataset'
        df['dataset'] = dataset.upper()

        all_df.append(df)
    
    all_df = pd.concat(all_df)
    all_df = all_df.reset_index()
    all_df = all_df.drop(columns=[0]) # drop the zero-shot
    origin_df = all_df.copy(deep=True)
    all_df = all_df.melt(id_vars=['dataset', 'size', 'config name', 'family'], var_name='# Samples', value_name='Test Loss') # melt the dataframe to a long format

    all_df['size'] = np.log(all_df['size']) # log scale
    all_df = all_df.rename(columns={'config name': 'Model'})

    # Select six models
    model_lst = [
        "cerebras/Cerebras-GPT-111M",
        "cerebras/Cerebras-GPT-256M",
        "cerebras/Cerebras-GPT-1.3B",
        "MBZUAI/LaMini-GPT-124M",
        "gpt2",
        "google/mt5-large",
    ]

    all_df = all_df[all_df['Model'].isin(model_lst)]

    g = sns.relplot(
        data=all_df,
        x="# Samples", 
        y="Test Loss",
        hue="Model",
        col="dataset",
        kind='scatter',
        legend='brief',
        alpha=0.5,
        aspect=1,
        facet_kws={'sharey': False, 'sharex': True}
    )
    

    # Plot laws
    handles, labels = g.axes[0][0].get_legend_handles_labels()

    for dataset in ['WMT19', 'GIGAWORD', 'FLAN']:
        cur_ax = None
        for ax in g.axes[0]:
            if ax.get_title().find(dataset) != -1:
                cur_ax = ax

        for model in model_lst:
            cur_color = handles[labels.index(model)].get_color()

            df = origin_df[origin_df['dataset'] == dataset]
            df = df.drop(columns=['size', 'family', 'dataset']) # drop unused columns
            df_model = df[df['config name'] == model].iloc[0]
            df_model = df_model.drop(labels=['config name'])

            x_plot = np.linspace(min(np.log(df_model.index.values.astype(float))), max(np.log(df_model.index.values.astype(float))), 100000)

            # fit our law
            fitted_params, bestloss = fit_our_law(np.log(df_model.index.values.astype(float)), np.log(df_model.to_numpy(dtype=np.float32)))

            # plot our law
            y_plot = our_law_transform((x_plot), *fitted_params)
            cur_ax.plot(np.exp(x_plot), np.exp(y_plot), color=cur_color, linewidth=1, linestyle='-')

            # fit vanilla law
            fitted_params, bestloss = fit_vanilla_law(np.log(df_model.index.values.astype(float)), np.log(df_model.to_numpy(dtype=np.float32)))

            # plot vanilla law
            y_plot = vanilla_law_transform((x_plot), *fitted_params)
            cur_ax.plot(np.exp(x_plot), np.exp(y_plot), color=cur_color, linewidth=1, linestyle='--')

    g.set_titles("{col_name}")

    # set log scale
    g.axes[0][0].set_xscale('log')
    g.axes[0][0].set_yscale('log')
    g.axes[0][1].set_yscale('log')
    g.axes[0][2].set_yscale('log')

    # set ticks
    g.axes[0][0].set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    g.axes[0][1].set_yticks([1.5, 2.0, 2.5, 3.0])
    g.axes[0][2].set_yticks([2.0, 2.5, 3.0, 3.5, 4.0])
    g.axes[0][0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    g.axes[0][1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    g.axes[0][2].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


    g.savefig('results/fig3.png', dpi=800)


def fig4():
    sns.set_theme(style="whitegrid")

    cache_path = 'results/rmsd.csv'
    if os.path.exists(cache_path):
        res = pd.read_csv(cache_path)
    else:
        res = []

        for dataset in ['wmt19', 'gigaword', 'flan']:
            df = load_benchmark(dataset)
            df.drop(columns=[0, 'size', 'family'], inplace=True)

            for model in df.index:
                df_model = df.loc[model]

                train_x = np.log(df_model.index.values.astype(float))
                train_y = np.log(df_model.to_numpy(dtype=np.float32))

                # fit our law
                fitted_params, bestloss = fit_our_law(train_x, train_y)
                pred_y = our_law_transform(train_x, *fitted_params)

                # calculate the RMSD under log scale
                our_rmsd_under_log = np.sqrt(np.mean((pred_y - train_y)**2))

                # fit original law
                fitted_params, bestloss = fit_vanilla_law(train_x, train_y)
                pred_y = vanilla_law_transform(train_x, *fitted_params)

                # calculate the RMSD under log scale
                vanilla_rmsd_under_log = np.sqrt(np.mean((pred_y - train_y)**2))

                res.append({
                    'dataset': dataset.upper(),
                    'model': model,
                    'our_rmsd_under_log': our_rmsd_under_log,
                    'vanilla_rmsd_under_log': vanilla_rmsd_under_log,
                })
        
        res = pd.DataFrame(res)
        # save the result
        res.to_csv(cache_path, index=False)


    g = sns.relplot(
        data=res, kind="scatter",
        y="our_rmsd_under_log", x="vanilla_rmsd_under_log", hue="dataset",
        aspect=2,
        s=100,
        alpha=0.7
    )

    g.set_xlabels("RMSD under log scale (Ours)")
    g.set_ylabels("RMSD under log scale (Vanilla)")

    # mean error of our law
    our_mean_error = np.mean(res['our_rmsd_under_log'])
    vanilla_mean_error = np.mean(res['vanilla_rmsd_under_log'])
    print(f'Our mean error: {our_mean_error}')
    print(f'Vanilla mean error: {vanilla_mean_error}')

    # plot the diagonal line
    g.axes[0][0].plot([-0.1, 0.5], [-0.1, 0.5], color='black', linewidth=1, linestyle='--')
    g.axes[0][0].set_xlim([-0.01, 0.14])
    g.axes[0][0].set_ylim([-0.01, 0.14])


    plt.savefig('results/fig4.png', dpi=300)


    our_rmsd_log_res = res.pivot(index='model', columns='dataset', values='our_rmsd_under_log')
    vanilla_rmsd_log_res = res.pivot(index='model', columns='dataset', values='vanilla_rmsd_under_log')

    # use the two dataframe to a new dataframe with columns [dataset1-our, dataset1-origin, dataset2-our, dataset2-origin, ...]
    res = pd.concat([our_rmsd_log_res, vanilla_rmsd_log_res], axis=1)
    
    #rename the columns
    res.columns = ['our-FLAN', 'our-GIGAWORD', 'our-WMT19', 'vanilla-FLAN', 'vanilla-GIGAWORD', 'vanilla-WMT19']

    # calculate the delta of RMSD
    res['delta-FLAN'] = res['vanilla-FLAN'] - res['our-FLAN']
    res['delta-WMT19'] = res['vanilla-WMT19'] - res['our-WMT19']
    res['delta-GIGAWORD'] = res['vanilla-GIGAWORD'] - res['our-GIGAWORD']

    # reorder columns
    res = res[['our-FLAN', 'vanilla-FLAN', 'delta-FLAN', 'our-WMT19', 'vanilla-WMT19', 'delta-WMT19', 'our-GIGAWORD', 'vanilla-GIGAWORD', 'delta-GIGAWORD']]

    with open('results/tab3.txt', 'w') as f:
        f.write(res.to_latex(float_format="%.4f", multirow=True))
    

def tab2():
    methods = [ATS_METHOD, OUR_LAW_METHOD, VANILLA_LAW_METHOD]
    datasets = ['FLAN', 'WMT19', 'GIGAWORD']
    ratios = ['1/8', '1/16', '1/32', '1/64', '1/128', '1/256', '1/512']
    metrics = [PEARSON_METRIC]
    # metrics = [RELACC_METRIC]

    cache_path = 'results/different_law_for_selection.csv'
    if os.path.exists(cache_path):
        res = pd.read_csv(cache_path)
    else:
        res = selection_performance(method_lst=methods)
        res.to_csv(cache_path, index=False)

    res = res.drop(columns=['max_data_num', 'predict_data_num'])

    # rename ratio to a string
    res['ratio'] = '1/' + res['ratio'].astype(str)

    # avg along (dataset, method, ratio, metric)
    res = res.groupby(['dataset', 'method', 'ratio', 'metric']).mean().reset_index()


    # remove other methods
    res = res[res['method'].isin(methods)]

    # group by (ratio, dataset, metric), and rank each method within each group
    res['rank'] = res.groupby(['dataset', 'ratio', 'metric'])['value'].rank(ascending=False, method='min')


    """Full table"""
    # create the table
    table = pd.DataFrame(columns=[dataset + '-' + method for dataset in datasets for method in methods], 
                         index=['\\textbf{' + metric + '}-' +  ratio for metric in metrics for ratio in ratios ],
                         dtype=object)

    for dataset in datasets:
        for method in methods:
            for metric in metrics:
                for ratio in ratios:
                    row = res[(res['dataset'] == dataset) & 
                            (res['method'] == method) & 
                            (res['ratio'] == ratio) &
                            (res['metric'] == metric)]
                    value = row['value'].values[0]
                    value_str = '{:.1f}'.format(value * 100)
                    # bold the best one, underline the second best one
                    if row['rank'].values[0] == 1:
                        cur_str = '\\textbf{' + value_str + '}'
                    elif row['rank'].values[0] == 2:
                        cur_str = '\\underline{' + value_str + '}'
                    else:
                        cur_str = value_str
                
                    table.loc['\\textbf{' + metric + '}-' + ratio, dataset + '-' + method] = cur_str

    # dump the table into a latex table
    with open('results/tab2.txt', 'w') as f:
        f.write(table.to_latex(float_format="%.4f", multirow=True))



def tab1():
    methods = [ATS_METHOD, ZERO_METHOD, SUBTUNING_METHOD, SIZE_METHOD]
    datasets = ['FLAN', 'WMT19', 'GIGAWORD']
    ratios = ['1/8', '1/16', '1/32', '1/64', '1/128', '1/256', '1/512']
    metrics = [PEARSON_METRIC, RELACC_METRIC]

    res = selection_performance(method_lst=methods)

    res = res.drop(columns=['max_data_num', 'predict_data_num'])
    res['ratio'] = '1/' + res['ratio'].astype(str)

    # avg along (dataset, method, ratio, metric)
    res = res.groupby(['dataset', 'method', 'ratio', 'metric']).mean().reset_index()

    # group by (ratio, dataset, metric), and rank each method within each group
    res['rank'] = res.groupby(['dataset', 'ratio', 'metric'])['value'].rank(ascending=False, method='min')


    """Full table"""
    # create the table
    table = pd.DataFrame(columns=[dataset + '-' + method for dataset in datasets for method in methods], 
                         index=['\\textbf{' + metric + '}-' +  ratio for metric in metrics for ratio in ratios ],
                         dtype=object)

    for dataset in datasets:
        for method in methods:
            for metric in metrics:
                for ratio in ratios:
                    row = res[(res['dataset'] == dataset) & 
                            (res['method'] == method) & 
                            (res['ratio'] == ratio) &
                            (res['metric'] == metric)]
                    value = row['value'].values[0] * 100
                    value_str = '{:.1f}'.format(value).lstrip('0')
                    # bold the best one, underline the second best one
                    if row['rank'].values[0] == 1:
                        cur_str = '\\textbf{' + value_str + '}'
                    elif row['rank'].values[0] == 2:
                        cur_str = '\\underline{' + value_str + '}'
                    else:
                        cur_str = value_str
                
                    table.loc['\\textbf{' + metric + '}-' + ratio, dataset + '-' + method] = cur_str

    # dump the table into a latex table
    with open('results/tab1.txt', 'w') as f:
        f.write(table.to_latex(float_format="%.4f", multirow=True))


def fig5():
    sns.set_theme(style="whitegrid")

    # Select six models
    model_lst = [
        "cerebras/Cerebras-GPT-111M",
        "cerebras/Cerebras-GPT-256M",
        "MBZUAI/LaMini-GPT-124M",
        "gpt2",
    ]

    df = load_benchmark('flan')

    """Plot the scatter plot of six models"""
    all_df = df
    all_df = all_df.reset_index()
    zeroshot_performance = all_df[[0, 'config name']].copy(deep=True)
    all_df = all_df.drop(columns=[0]) # drop the zero-shot
    origin_df = all_df.copy(deep=True)
    all_df = all_df.melt(id_vars=['size', 'config name', 'family'], var_name='# Samples', value_name='Test Loss') # melt the dataframe to a long format
    all_df['size'] = np.log(all_df['size']) # log scale
    all_df = all_df.rename(columns={'config name': 'Model'})

    all_df = all_df[all_df['Model'].isin(model_lst)]

    g = sns.relplot(
        data=all_df,
        x="# Samples", 
        y="Test Loss",
        hue="Model",
        kind='scatter',
        legend='brief',
        alpha=0.5,
        aspect=1.5,
    )
    

    """Plot laws"""
    handles, labels = g.axes[0][0].get_legend_handles_labels()

    df = df.reset_index()
    df = df.drop(columns=['size', 'family', 0]) # drop unused columns

    cur_ax = g.axes[0][0]
    for model in model_lst:
        cur_color = handles[labels.index(model)].get_color()

        df_model = df[df['config name'] == model].iloc[0]
        df_model = df_model.drop(labels=['config name'])

        x_plot = np.linspace(min(np.log(df_model.index.values.astype(float))), max(np.log(df_model.index.values.astype(float))), 100000)

        # fit our law
        fitted_params, bestloss = fit_our_law(np.log(df_model.index.values.astype(float)), np.log(df_model.to_numpy(dtype=np.float32)))
        y_plot = our_law_transform((x_plot), *fitted_params)

        # plot our law
        cur_ax.plot(np.exp(x_plot), np.exp(y_plot), color=cur_color, linewidth=1, linestyle='-')

        # draw zeroshot performance of current model
        zeroshot = zeroshot_performance[zeroshot_performance['config name'] == model].iloc[0][0]
        cur_ax.plot(np.linspace(50, 1e7, 100000), [zeroshot] * 100000, color=cur_color, linewidth=1, linestyle='--')
        

    # set log scale
    g.axes[0][0].set_xscale('log')
    g.axes[0][0].set_yscale('log')

    # set ticks
    g.axes[0][0].set_yticks([2.0, 2.5, 3.0, 3.5, 4.0])
    g.axes[0][0].set_xlim([150, 2e6])
    g.axes[0][0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    g.savefig('results/fig5.png', dpi=200)


def fig11():
    dataset_lst=['flan', 'wmt19', 'gigaword']
    ratio_lst=[8, 16, 32, 64, 128, 256, 512]
    sns.set_theme(font_scale=1.5)
    cache_path = 'results/ablation_study_hyper.csv'
    if os.path.exists(cache_path):
        res = pd.read_csv(cache_path)
    else:
        res = []
        for dataset_name in dataset_lst:
            df = load_benchmark(dataset_name)
            for k in [3, 4, 5]:
                for delta in [1,2,3,4,5]:
                    for max_data_num in [204800, 102400, 51200, 25600, 12800, 6400, 3200]:
                        for ratio in ratio_lst:
                            predict_data_num = max_data_num * ratio
                            if predict_data_num != 1638400:
                                continue
                            _, ats_feature = ats_select(df, max_data_num, predict_data_num, k=k, delta=delta)
                            _, _, _, golden_performance = golden_select(df, predict_data_num)
                            ats_pearson_corr = round(pearsonr(ats_feature, golden_performance)[0], 4)

                            res.append({'dataset': dataset_name.upper(),
                                        'ratio': ratio,
                                        'max_data_num': max_data_num,
                                        'predict_data_num': predict_data_num,
                                        'k': k,
                                        'delta': delta,
                                        'metric': PEARSON_METRIC,
                                        'value': ats_pearson_corr})
        
        res = pd.DataFrame(res)

        res.to_csv(cache_path, index=False)

    res['value'] = res.apply(lambda x: round(x['value'] * 100, 2), axis=1)
    res = res.drop(columns=['max_data_num', 'predict_data_num'])

    # average over (dataset, ratio, k, delta, metric)
    res = res.groupby(['dataset', 'ratio', 'k', 'delta', 'metric']).mean().reset_index()


    """Draw Heatmap"""
    # create 3*7 subplots
    fig, axs = plt.subplots(len(dataset_lst), 7, figsize=(30, 20))

    cmap = sns.color_palette("light:white", as_cmap=True)


    for i, dataset_name in enumerate(dataset_lst):
        dataset_name = dataset_name.upper()
        for j, ratio in enumerate([8, 16, 32, 64, 128, 256, 512]):
            cur_ax = axs[i][j] if len(dataset_lst) > 1 else axs[j]
            df = res[(res['dataset'] == dataset_name) & (res['ratio'] == ratio) & (res['metric'] == PEARSON_METRIC)]
            df = df.pivot(columns='k', index='delta', values='value')
            sns.heatmap(df, annot=True, ax=cur_ax, vmin=0.5, vmax=0.5, cmap=cmap, fmt='.2f', cbar=False,
                        linewidths=1, linecolor='white')
            cur_ax.set_ylabel('')
            cur_ax.set_xlabel('')

            if j != 0:
                cur_ax.set_yticks([])
            if i != len(dataset_lst) - 1:
                cur_ax.set_xticks([])

            cur_ax.set_title(dataset_name + ' ($\\gamma$=1/' + str(ratio)+')')

    axs[1][0].set_ylabel('$\\delta$', fontsize=40) 
    axs[2][3].set_xlabel('K', fontsize=40)

    plt.tight_layout()
    
    plt.savefig('results/fig11.png', dpi=300)


def fig12():
    size_constraints = [
        7e9,
        1.9e9,
        1.4e9,
        700e6,
    ]
    
    cache_path = 'results/ablation_size_constraints.csv'
    if os.path.exists(cache_path):
        res = pd.read_csv(cache_path)
    else:
        res_lst = []
        for constratin in size_constraints:
            res = selection_performance(method_lst=[ATS_METHOD, ZERO_METHOD, SUBTUNING_METHOD, SIZE_METHOD],
                                        size_constraint=constratin)
            res['size_constraint'] = constratin
            res_lst.append(res)
        
        res = pd.concat(res_lst)

        res.to_csv(cache_path, index=False)
    
    res['method'] = res['method'].apply(lambda x: 'AtS' if x == ATS_METHOD else x)
    res['ratio'] = res['ratio'].apply(lambda x: '1/' + str(x))
    res = res[res['metric'] == PEARSON_METRIC]
    res = res.drop(columns=['max_data_num', 'predict_data_num', 'metric'])

    # draw a catplot
    sns.set_theme(style="whitegrid")

    palette = sns.color_palette(["#ed6056", "#56b66f", "#5c8cda", "#fbc62c"], 4)
    palette = palette + sns.color_palette()[4:] # concat with default color palette

    sns.set_theme(font_scale=1.5) 
    with sns.color_palette(palette):
        g = sns.catplot(
            data=res, kind="bar",
            x="ratio", 
            y="value", 
            hue="method",
            row='size_constraint', 
            errorbar="sd",
            height=6,
            aspect=4,
        )

    g.set_ylabels('Pearson Correlation')

    g.axes[0][0].set_title("Model Size < 7B")
    g.axes[1][0].set_title("Model Size < 2B")
    g.axes[2][0].set_title("Model Size < 1.4B")
    g.axes[3][0].set_title("Model Size < 700M")


    plt.savefig('results/fig12.png', dpi=300)


def fig7():
    methods = [ATS_METHOD, ZERO_METHOD, SUBTUNING_METHOD, SIZE_METHOD]
    metrics = [PEARSON_METRIC]

    res = selection_performance(method_lst=methods)
    res = res.drop(columns=['max_data_num', 'predict_data_num'])

    # avg along (dataset, method, ratio, metric)
    res = res.groupby(['dataset', 'method', 'ratio', 'metric']).mean().reset_index()


    # remove other methods
    res = res[res['method'].isin(methods) & res['metric'].isin(metrics)]

    dataset2sampleLen = {
        'FLAN': 23 + 12,
        'WMT19': 32 + 40,
        'GIGAWORD': 36 + 8
    }
    max_data_num = 1638400
    epoch_num = 20
    model_param_num = 32837*10**6
    random_num = 3
    hyper_search_num = 12
    
    def compute_flops_func(row):
        if row['method'] == ATS_METHOD:
            sample_length = dataset2sampleLen[row['dataset']]
            dataset_num = sum([200 * 2**i for i in range(20) if 200 * 2**i <= max_data_num // row['ratio']])
            token_num = sample_length * dataset_num * epoch_num
            flops = 6 * token_num * model_param_num
            total_flops = flops * (random_num + hyper_search_num)
        
        elif row['method'] == SUBTUNING_METHOD:
            sample_length = dataset2sampleLen[row['dataset']]
            dataset_num = max_data_num // row['ratio'] 
            token_num = sample_length * dataset_num * epoch_num
            flops = 6 * token_num * model_param_num
            total_flops = flops * (random_num + hyper_search_num)
        
        elif row['method'] == SIZE_METHOD:
            total_flops = 0
        
        elif row['method'] == ZERO_METHOD:
            total_flops = 0

        return total_flops

    res['flops'] = res.apply(compute_flops_func, axis=1)
    res = pd.concat(
        (res,
        pd.DataFrame([
            {'dataset': 'FLAN', 'method': 'FullTuning', 'ratio': 1/8, 'metric': 'PEARSON', 'value': 1, 'flops': dataset2sampleLen['FLAN'] * max_data_num * epoch_num * model_param_num * 6 * (random_num + hyper_search_num)},
            {'dataset': 'WMT19', 'method': 'FullTuning', 'ratio': 1/8, 'metric': 'PEARSON', 'value': 1, 'flops': dataset2sampleLen['WMT19'] * max_data_num * epoch_num * model_param_num * 6 * (random_num + hyper_search_num)},
            {'dataset': 'GIGAWORD', 'method': 'FullTuning', 'ratio': 1/8, 'metric': 'PEARSON', 'value': 1, 'flops': dataset2sampleLen['GIGAWORD'] * max_data_num * epoch_num * model_param_num * 6 * (random_num + hyper_search_num)},
        ])
        )
    )

    # rename 
    res.loc[res['method'] == ATS_METHOD, 'method'] = 'AtS'

    # draw a scatter plot
    sns.set_theme(style="whitegrid")

    g = sns.relplot(
        data=res, kind="scatter",
        x="flops", y="value", hue="method",
        col='dataset', 
        aspect=1.0,
        alpha=0.7,
        s=100
    )

    g.set_xlabels("FLOPS")
    g.set_ylabels("Pearson Correlation")

    g.savefig('results/fig7.png', dpi=300)


if __name__ == '__main__':
    fig1_a()
    fig1_b()
    fig3()
    fig4()
    tab1()
    tab2()
    fig5()
    fig7()
    fig11()
    fig12()