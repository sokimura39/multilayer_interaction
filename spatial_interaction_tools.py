# libraries
import pandas as pd
import scipy
import math
import numpy as np
import networkx as nx


# calculate the Oi and Dj
def calc_OiDj(df: pd.DataFrame, source = 'source', target = 'target', value = 'flows'):
    '''
    Calculate Oi and Dj for each origin and destination and merge back into data frame

    Parameters
    ----------
    df : pd.DataFrame 
        OD matrix
    source, target, value : str
        column names

    Returns
    -------
    df_edit : pd.DataFrame
        OD matrix with the columns Oi and Dj attached
    '''
    # copy data frame
    df_edit = df.copy()

    # create Oi and Dj dfs
    Oi = df_edit.rename(columns = {value: 'Oi'}).groupby(source)['Oi'].sum()
    Dj = df_edit.rename(columns = {value: 'Dj'}).groupby(source)['Dj'].sum()
    
    # merge with original dataframe
    df_edit = df_edit.merge(
        Oi,
        how = 'left',
        left_on = source,
        right_index = True
    ).merge(
        Dj,
        how = 'left',
        left_on = 'target',
        right_index = True
    )

    # return df
    return df_edit

# calculate average distance
def calculate_cbar(df: pd.DataFrame, flow: str, cost: str):
    '''
    Calculates the average distance travelled using the mode.

    Parameters
    ---------
    df: pandas.DataFrame
        dataframe of the origin-destination matrix
    volume: str
        column name for the number of flows for each OD pair
    distance: str
        column name for the cost between each OD pair

    Returns
    -------
    cbar: float
        the mean distance travelled on the mode 
    '''
    # calculate cbar
    cbar = (df[cost] * df[flow]).sum() / df[flow].sum()

    return cbar

# define functions for testing
def CalcRSquared(observed, estimated):
    """Calculate the r^2 from a series of observed and estimated target values
    inputs:
    Observed: Series of actual observed values
    estimated: Series of predicted values"""
    
    r, p = scipy.stats.pearsonr(observed, estimated)
    R2 = r **2
    
    return R2

def CalcRMSE(observed, estimated):
    """Calculate Root Mean Square Error between a series of observed and estimated values
    inputs:
    Observed: Series of actual observed values
    estimated: Series of predicted values"""
    
    res = (observed -estimated)**2
    RMSE = round(math.sqrt(res.mean()), 3)
    
    return RMSE

def calibrate_model(od_matrix: pd.DataFrame, modes: dict, thres = 0.0005, source = 'source', target = 'target', calib_beta = True, beta_init = 1.0, AiBj_init = 1.0, show_log = True):
    '''
    Calibrate the doubly constrained model and return the parameters and the predicted results.

    Parameters
    ----------
    od_matrix : pandas.DataFrame
        dataframe of the od datarframe
    modes : dict
        dictionary containing the modes and column names for the cost, path, and observed flows.
    thres : float
        the threshold to which the model is recognised as converging.
    source : str
        column name for source
    target : str
        column name for target
    calib_beta : bool
        flag to calibrate beta or not. If true, calibrates the cost function. 
        If false, simply calculate the beta based on the given betas
    beta_init : float or dict
        initial values for beta. If dict is provided, use as initial values.
    AiBj_init : float
        initial values for Ai and Bj
    show_log : bool
        print log if true

    Returns
    -------
    beta : dict
        dictionary of parameters
    '''
    # get number of modes
    num_modes = len(modes['modes'])
    modes_list = modes['modes']

    # print what we will be doing
    if show_log:
        print(f'Simulating for {num_modes} modes: {modes_list}')

    # ----- data preparation -----

    # initialise beta
    beta = {}
    # if dict is provided use
    if isinstance(beta_init, dict):
        beta = beta_init
    # if float is provided use that
    else:
        for m in modes_list:
            beta[m] = beta_init

    # get the required columns and copy df
    columns = [source, target] + modes['observed_flow'] + modes['cost']
    if 'path' in modes:
        columns += modes['path']
    df = od_matrix[columns].copy()

    # calculate total flows
    df.insert(num_modes + 2, 'obs_total_flow', df[modes['observed_flow']].sum(axis = 1))

    # calculate Oi and Dj
    df = calc_OiDj(df, value = 'obs_total_flow')

    # set converged to false
    converged_beta = False

    # ----- calculate Ai and Bj -----
    df_temp = df.copy()

    # initialise Ai and Bj
    df_temp['Ai'] = AiBj_init
    df_temp['Bj'] = AiBj_init

    # count iteration
    iteration = 0

    while not converged_beta:
        # increment calibration if beta is to be calibrated
        if calib_beta:
            iteration += 1
            if show_log:
                print(f'Iteration {iteration}')

        converged_AB = False
        iteration_AB = 0
        while not converged_AB:
            iteration_AB += 1
            # calculate new Ai
            for i in range(num_modes):
                # get the mode
                mode = modes_list[i]
                df_temp[f'BjDjexp_{mode}'] = df_temp['Bj'] * df_temp['Dj'] * np.exp(-beta[mode] * df_temp[modes['cost'][i]])
            

            columns_Ai = [c for c in df_temp.columns if 'BjDjexp_' in c]
            Ai_df = df_temp.groupby(source)[columns_Ai].sum()
            Ai_df['Ai_new'] = 1 / Ai_df.sum(axis = 1)

            # calculate the difference between old and new values
            Ai_df['Ai'] = df_temp.groupby(source)['Ai'].mean()
            Ai_df['diff'] = abs((Ai_df['Ai'] - Ai_df['Ai_new']) / Ai_df['Ai'])

            # merge back to the original dataframe
            df_temp = df_temp.drop(columns = 'Ai').merge(
                Ai_df['Ai_new'],
                how = 'left',
                left_on = source,
                right_index = True
            ).rename(columns = {'Ai_new': 'Ai'})

            # calculate new Bj
            # make sure we do A then B
            for i in range(num_modes):
                # get the mode
                mode = modes_list[i]
                df_temp[f'AiOiexp_{mode}'] = df_temp['Ai'] * df_temp['Oi'] * np.exp(-beta[mode] * df_temp[modes['cost'][i]])
            
            columns_Bj = [c for c in df_temp.columns if 'AiOiexp_' in c]
            Bj_df = df_temp.groupby(target)[columns_Bj].sum()
            Bj_df['Bj_new'] = 1 / Bj_df.sum(axis = 1)
            
            # calculate the difference between old and new values
            Bj_df['Bj'] = df_temp.groupby(target)['Bj'].mean()
            Bj_df['diff'] = abs((Bj_df['Bj'] - Bj_df['Bj_new']) / Bj_df['Bj'])

            # merge back to the original dataframe
            df_temp = df_temp.drop(columns = 'Bj').merge(
                Bj_df['Bj_new'],
                how = 'left',
                left_on = target,
                right_index = True
            ).rename(columns = {'Bj_new': 'Bj'})
    
            # check if converged
            if max(Bj_df['diff'].sum(), Ai_df['diff'].sum()) < 0.5:
                converged_AB = True

        if show_log:
            print(f'Iterations for calibrating Ai, Bj: {iteration_AB}')

        # ----- calculate prediction -----
        # initialise cbar
        cbar_dict = {
            'modes': [],
            'observed': [],
            'predicted': []
        }

        for i in range(num_modes):
            # get the mode
            mode = modes_list[i]
            # calculate the modes
            df_temp[f'pred_{mode}'] = df_temp['Ai'] * df_temp['Bj'] * df_temp['Oi'] * df_temp['Dj'] * np.exp(-beta[mode] * df_temp[modes['cost'][i]])
            # calculate cbar
            cbar_dict['modes'].append(mode)
            cbar_dict['observed'].append(calculate_cbar(df_temp, modes['observed_flow'][i], modes['cost'][i]))
            cbar_dict['predicted'].append(calculate_cbar(df_temp, f'pred_{mode}', modes['cost'][i]))

        # calibrate the beta if required
        converged_beta = True

        if calib_beta:
            for i in range(num_modes):
                # get the mode
                mode = cbar_dict['modes'][i]
                delta = abs(cbar_dict['observed'][i] - cbar_dict['predicted'][i])

                if delta / cbar_dict['observed'][i] > thres:
                    converged_beta = False
                    beta[mode] = beta[mode] * cbar_dict['predicted'][i] / cbar_dict['observed'][i]
                    if show_log:
                        print(f'{mode} not converged. New beta: {beta[mode]}')
                else:
                    if show_log:
                        print(f'{mode} converged. beta: {beta[mode]}')
    
    # calculate total
    pred_columns = [c for c in df_temp.columns if 'pred_' in c]
    df_temp['pred_total'] = df_temp[pred_columns].sum(axis = 1)

    columns_df = [source, target, 'Ai', 'Bj'] + [c for c in df_temp.columns if 'pred_' in c]

    # merge once converged
    df = df.merge(
        df_temp[columns_df],
        how = 'left',
        on = [source, target]
    )

    if show_log:
        print('Finished.')

    # only return the df when beta not calibrated, but return both if it is
    if calib_beta:
        return df, beta
    else:
        return df

def calculate_flow(G: nx.MultiDiGraph, od_matrix: pd.DataFrame, flow: str, path: str, attr_name = 'flows'):
    '''
    Calculate the Flows on each segment of the network.

    Parameters
    ----------
    G : nx.MultiDiGraph
        the network that we want to add the flows
    od_matrix : pd.DataFrame
        the od matrix data to use
    flow : str
        column name of the flow
    path : str
        column name of the path
    attr_name : str
        the name of attribute to add to the network

    Returns
    -------
    G_edit : nx.MultiDiGraph
        network with the attributes added
    '''

    # initialise flow dict
    flow_dict = {}
    for idx, row in od_matrix.iterrows():
        flows = row[flow]
        path_list = row[path]



        if isinstance(path_list, list) and (len(path_list) >= 2):
            for u, v in zip(path_list[:-1], path_list[1:]):
                if (u, v) in flow_dict:
                    flow_dict[(u, v, 0)] += flows
                else:
                    flow_dict[(u, v, 0)] = flows
    
    # copy network
    G_edit = G.copy()
    nx.set_edge_attributes(G_edit, flow_dict, attr_name)

    for u,v,k,d in G_edit.edges(data = True, keys = True):
        if attr_name not in d:
            d[attr_name] = 0

    return G_edit       