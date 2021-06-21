def data_analysis(link, lane, training_folder, test_folder, variables):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import numpy as np
    import seaborn as sn
    import pickle
    import glob
    import os
    import random
    pd.set_option('display.max_columns', 8)

    ## Load data
    folder = training_folder
    training_files = glob.glob(os.path.join(folder, '*.csv'))
    list_files = [x for x in training_files if str(int(link))+str(int(lane)) in x]
    #random.shuffle(list_files)
    counter = 0
    counter2 = 0
    for file in list_files:
        if 'data' not in globals():
            data = pd.read_csv(file, header = 0)
            counter += 1
        else:
            data_new = pd.read_csv(file,header = 0)
            data_new['Current_cycle'] = data_new['Current_cycle'].astype(int) + counter * 18000
            data = pd.concat([data, data_new])
            counter += 1

    ## Plot boxplot per variable
    # for variable in variables:
    #     a = data[[variable, 'State']]
    #     ax = a.boxplot(by='State', meanline=True, showmeans=True, showcaps=True, showbox=True,
    #                     showfliers=True, return_type='axes')
    #     plt.show()
    #
    # ## Plot distplot per variable
    # def plotchart(col):
    #     fix, (ax1,ax2) = plt.subplots(1,2,figsize=(7,5))
    #     sn.boxplot(col, orient='v', ax = ax1)
    #     ax1.set_ylabel=col.name
    #     ax1.set_title('Box Plot of {}'.format(col.name))
    #     sn.distplot(col,ax=ax2)
    #     ax2.set_title('Distribution plot of {}'.format(col.name))
    #     plt.show()
    #
    # def analysis_column(col):
    #     print('Mean',format(col.mean()))
    #     print('Median',format(col.median()))
    #     plotchart(col)
    #
    # for variable in variables:
    #     print(variable)
    #     analysis_column(data[variable])
    # ## Number of states
    # counts = data.State.value_counts()
    # print(counts)
    # ## Correlation
    # corr = data.corr(method='spearman')
    # plt.figure(figsize=(15,15))
    # sn.heatmap(corr, vmax = .8, linewidths=0.01, square = True, annot=True, cmap= 'RdBu', linecolor='black')
    # plt.show()
    #
    # sn.pairplot(data[variables+['State',]], hue = 'State', diag_kind='kde',
    #             palette = {0:'blue', 1:'green', 2:'orange', 3:'red', 4:'purple'}, markers = '.')
    # plt.show()
    # ## Density functions
    # data_vars = data[variables+['State']]
    # fig = plt.figure(figsize = (30,20))

    # j = 0
    # for i in variables:
    #     plt.subplot(1,6, j+1)
    #     j+=1
    #     sn.distplot(data_vars.loc[data_vars.State == 0][i], color = 'r', hist =False, label = 'undersaturated')
    #     sn.distplot(data_vars.loc[data_vars.State == 1][i], color = 'b',hist =False, label = 'O1')
    #     sn.distplot(data_vars.loc[data_vars.State == 2][i], color = 'orange', hist =False,label = 'O2')
    #     sn.distplot(data_vars.loc[data_vars.State == 3][i], color='g',hist =False, label='O3')
    #     sn.distplot(data_vars.loc[data_vars.State == 4][i], color='y', hist =False,label='Spillover')
    #     plt.legend()
    # fig.suptitle('Data analysis')
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.95)
    # plt.show()

    # # MULTICOLLINEARITY #https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
    data_vars = data[variables]
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    VIF = pd.DataFrame()
    VIF['Columns'] = data_vars.columns
    VIF['VIF'] = [variance_inflation_factor(data_vars.values,i) for i in range(len(data_vars.columns))]
    print('VIF', VIF)

    lst_data = data['Min_stop'].to_list()
    for i in range(len(lst_data)):
        if lst_data[i] == 0:
            lst_data[i] += 20
    data['Min_stop'] = lst_data
    # Linear relation to log-odds
    import seaborn as sns
    data_adapt = data.copy()
    dummy = pd.get_dummies(data_adapt['State'])
    data_adapt = pd.concat([data_adapt,dummy], axis = 1)
    data_adapt = data_adapt.rename({0:'U',1: 'O1', 2:'O2',3:'O3', 4:'Sp'},axis=  1)
    for var in variables:
        x = sns.regplot(x=var, y='Sp', data = data_adapt, logistic=True).set_title(str(var)+' Log Odds Plot')
        plt.show()