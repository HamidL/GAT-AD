from multiprocessing import get_context
from random import randint
from datetime import (
    datetime,
    timedelta
)
from dateutil import parser
import re

import torch
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw
from sklearn.preprocessing import MinMaxScaler


def abilene_topology(OSPF=True):
    graph = nx.Graph()
    links = ['KSCYng', 'NYCMng', 'ATLAM5', 'ATLAng', 'DNVRng', 'IPLSng', 'HSTNng', 'STTLng', 'CHINng', 'WASHng',
             'SNVAng', 'LOSAng']
    for link in links:
        graph.add_node(link,
                       node_type="node")

    # generate all links possible (30) and set order for homogenization
    edges = ['ATLAng ATLAM5', 'HSTNng ATLAng', 'IPLSng ATLAng', 'WASHng ATLAng', 'IPLSng CHINng', 'NYCMng CHINng',
             'KSCYng DNVRng', 'SNVAng DNVRng', 'STTLng DNVRng', 'KSCYng HSTNng', 'LOSAng HSTNng', 'KSCYng IPLSng',
             'SNVAng LOSAng', 'WASHng NYCMng', 'STTLng SNVAng']
    edges_OSPF = ['ATLAng ATLAM5 1', 'HSTNng ATLAng 1176', 'IPLSng ATLAng 587', 'WASHng ATLAng 846',
                  'IPLSng CHINng 260', 'NYCMng CHINng 700', 'KSCYng DNVRng 639', 'SNVAng DNVRng 1295',
                  'STTLng DNVRng 2095', 'KSCYng HSTNng 902', 'LOSAng HSTNng 1893', 'KSCYng IPLSng 548',
                  'SNVAng LOSAng 366', 'WASHng NYCMng 233', 'STTLng SNVAng 861']
    if not OSPF:
        for edge in edges:
            origin, dest = edge.split(" ")
            graph.add_edge(origin, dest)
    else:
        for edge in edges_OSPF:
            origin, dest, OSPF_cost = edge.split(" ")
            graph.add_edge(origin, dest, weight=int(OSPF_cost))
    graph = graph.to_directed()
    line_graph = nx.line_graph(graph)
    line_links = dict()
    for id, link in enumerate(list(line_graph.nodes())):
        line_links[link] = id

    src_link_to_link, dst_link_to_link = [], []
    for node_from, node_to in line_graph.edges():
        src_link_to_link.append(line_links[node_from])
        dst_link_to_link.append(line_links[node_to])

    # generate all source-destination paths possible (132) and set order for homogenization
    od_combinations = []
    for source in links:
        for dest in links:
            if source != dest:
                od_combinations.append((source, dest))
    paths = dict()
    for id, path in enumerate(od_combinations):
        paths[path] = id

    # shortest path for each combination
    shortest_paths = dict()
    for path_source, path_dest in od_combinations:
        shortest_paths[(path_source, path_dest)] = nx.shortest_path(graph, path_source, path_dest)

    # links
    src_path_to_link, dst_link_to_path = [], []
    src_link_to_path, dst_path_to_link = [], []
    for path, links in shortest_paths.items():
        for i in range(len(links) - 1):
            src_path_to_link.append(paths[path])
            dst_link_to_path.append(line_links[(links[i], links[i+1])])
            src_link_to_path.append(line_links[(links[i], links[i+1])])
            dst_path_to_link.append(paths[path])

    path_to_link = (src_path_to_link, dst_link_to_path)
    link_to_path = (src_link_to_path, dst_path_to_link)
    link_to_link = (src_link_to_link, dst_link_to_link)

    return graph, line_links, paths, shortest_paths, path_to_link, link_to_path, link_to_link


def calculate_similarity(args):
    i, data_i, data = args
    node_similarity = []

    # Calculate DTW distance between the current node and all other nodes
    for j, data_j in enumerate(data):
        if i != j:  # Exclude self-similarity
            distance, _ = fastdtw(data_i, data_j)
            node_similarity.append((distance, j))

    # Sort the node similarity list based on DTW distance
    node_similarity.sort()
    return node_similarity


def compute_similarities(data):
    num_nodes = len(data)
    node_similarities = []

    with get_context("fork").Pool() as pool:
        args = [(i, data[i], data) for i in range(num_nodes)]
        node_similarities = list(tqdm(pool.imap(calculate_similarity, args), total=num_nodes))

    return node_similarities


def read_and_transform_WADI_2019(base_dir):
    normal_data = pd.read_csv(base_dir / "WADI.A2_19 Nov 2019/WADI_14days_new.csv")

    # Start handling of missing values
    na_counts = normal_data.isna().sum()
    columns_to_fill = na_counts[na_counts < 100].index
    columns_to_fill_with_zeros = na_counts[na_counts >= 100].index
    normal_data[columns_to_fill] = normal_data[columns_to_fill].fillna(normal_data[columns_to_fill].mean())
    normal_data[columns_to_fill_with_zeros] = normal_data[columns_to_fill_with_zeros].fillna(0)
    # End handling of missing values

    col_names = normal_data.columns[3:].tolist()
    col_ids = {col: i for i, col in enumerate(col_names)}
    src_node_to_node = []
    dst_node_to_node = []
    for src in range(len(col_names)):
        for dst in range(len(col_names)):
            if src != dst:
                src_node_to_node.append(src)
                dst_node_to_node.append(dst)
    node_to_node = torch.tensor((src_node_to_node, dst_node_to_node))
    normal_data = torch.tensor(normal_data.iloc[:, 3:].to_numpy().T).float()
    # normal_data = torch.tensor(MinMaxScaler().fit_transform(normal_data).clip(0, 1))
    # torch.save(normal_data, base_dir / "WADI_train.pth")

    abnormal_data = pd.read_csv(base_dir / "WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv", header=1)

    # Start handling of missing values
    abnormal_na_counts = abnormal_data.isna().sum()
    abnormal_columns_to_fill = abnormal_na_counts[abnormal_na_counts < 100].index
    abnormal_columns_to_drop = abnormal_na_counts[abnormal_na_counts >= 100].index
    abnormal_data[abnormal_columns_to_fill] = abnormal_data[abnormal_columns_to_fill].fillna(abnormal_data[abnormal_columns_to_fill].mean())
    abnormal_data = abnormal_data.drop(columns=abnormal_columns_to_drop)
    # End handling of missing values

    abnormal_label = abnormal_data.iloc[:, -1] == -1
    abnormal_label = torch.tensor(abnormal_label.to_numpy().astype(int))
    abnormal_data = torch.tensor(abnormal_data.iloc[:, 3:-1].to_numpy().T).float()
    # abnormal_data = torch.tensor(MinMaxScaler().fit_transform(abnormal_data).clip(0, 1))

    wadi_data = dict(
        train=normal_data,
        test=abnormal_data,
        test_labels=abnormal_label,
        node_to_node=node_to_node,
        node_indices=col_ids,
        comment="Original WADI dataset from 19 Nov 2019."
    )
    return wadi_data


def read_and_transform_WADI_2017_GDN(base_dir):
    """
    Credit: https://github.com/huankoh/CST-GL/blob/main/generate_data/generate_wadi_data.ipynb
    """
    train = pd.read_csv(base_dir / "WADI.A1_9 Oct 2017/WADI_14days.csv", header=3)
    train = train.fillna(0)
    train = train.drop(columns=['Row'])
    train['Date'] = train['Date'].apply(lambda x: '/'.join([i.zfill(2) for i in x.split('/')]))
    train['Time'] = train['Time'].apply(lambda x: x.replace('.000',''))
    train['Time'] = train['Time'].apply(lambda x: ':'.join([i.zfill(2) for i in x.split(':')]))
    train['datetime'] = train.apply(lambda x: datetime.strptime(x.Date +' '+x.Time,'%m/%d/%Y %I:%M:%S %p'),axis=1)
    assert train['datetime'].tolist() == train.apply(lambda x: parser.parse(x.Date +' '+x.Time,fuzzy=True), axis=1).tolist()
    # Rename and resort
    coi = ['datetime'] + [i for i in train.columns if i not in ['Date','Time','datetime']]
    train = train[coi]
    train = train.sort_values('datetime')
    empty_cols = [col for col in train.columns if train[col].isnull().all()]
    train[empty_cols] = train[empty_cols].fillna(0,inplace=True)
    for i in train.columns[train.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
        train[i].fillna(train[i].mean(),inplace=True)

    """
    MINMAX norm
    """
    scaler = MinMaxScaler()
    X_train = train.iloc[:,1:].to_numpy()
    scaler.fit(X_train)
    normalized_train_X = scaler.transform(X_train)

    normalized_train = train.copy()
    normalized_train.iloc[:,1:] = normalized_train_X

    """
    Downsample
    """
    filtered_train = normalized_train.groupby(np.arange(len(train))//10).median()
    time = [normalized_train.iloc[0,0] + timedelta(seconds=10*i) for i in range(0,len(filtered_train))]

    filtered_train['datetime'] = time
    filtered_train = filtered_train.iloc[2160:]
    filtered_train = filtered_train.iloc[:-1,:]
    filtered_train = filtered_train[coi]

    """
    Replace WIN strings
    """
    pat = re.escape('\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\')
    rename_col = [re.sub(pat,'',i) for i in filtered_train.columns]
    filtered_train.columns = rename_col
    final_train = filtered_train.reset_index().drop(columns=['datetime','index'])
    final_train = final_train.reset_index().rename(columns={'index':'timestamp'})

    attacks = []

    # Attack 1
    start = datetime(2017,10,9,19,25,00)
    end = datetime(2017,10,9,19,50,16)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]
    print(attacks[0],attacks[-1])
    # Attack 2
    start = datetime(2017,10,10,10,24,10)
    end = datetime(2017,10,10,10,34,00)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]

    # Attack 3-4
    start = datetime(2017,10,10,10,55,00)
    end = datetime(2017,10,10,11,24,00)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]
    #attacks['1_AIT_001'] = [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds()))]

    # Attack 5
    start = datetime(2017,10,10,11,30,40)
    end = datetime(2017,10,10,11,44,50)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]

    # Attack 6
    start = datetime(2017,10,10,13,39,30)
    end = datetime(2017,10,10,13,50,40)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]
    #attacks['2_MCV_101'] = [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds()))]

    # Attack 7
    start = datetime(2017,10,10,14,48,17)
    end = datetime(2017,10,10,14,59,55)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]


    # Attack 8
    start = datetime(2017,10,10,17,40,00)
    end = datetime(2017,10,10,17,49,40)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]


    # Attack 9
    start = datetime(2017,10,10,10,55,00)
    end = datetime(2017,10,10,10,56,27)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]

    # Attack 10
    start = datetime(2017,10,11,11,17,54)
    end = datetime(2017,10,11,11,31,20)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]


    # Attack 11
    start = datetime(2017,10,11,11,36,31)
    end = datetime(2017,10,11,11,47,00)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]


    # Attack 12
    start = datetime(2017,10,11,11,59,00)
    end = datetime(2017,10,11,12,5,00)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]


    # Attack 13
    start = datetime(2017,10,11,12,7,30)
    end = datetime(2017,10,11,12,10,52)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]

    # Attack 14
    start = datetime(2017,10,11,12,16,00)
    end = datetime(2017,10,11,12,25,36)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]

    # Attack 15
    start = datetime(2017,10,11,15,26,30)
    end = datetime(2017,10,11,15,37,00)

    attacks += [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]

    tt = [start + timedelta(seconds=1*i) for i in range(int((end-start).total_seconds())+1)]

    print(tt[0],tt[-1])

    attacks_set = set(attacks)

    test = pd.read_csv(base_dir / "WADI.A1_9 Oct 2017/WADI_attackdata.csv")
    #test = test.fillna(0)
    test = test.drop(columns=['Row'])
    test['Date'] = test['Date'].apply(lambda x: '/'.join([i.zfill(2) for i in x.split('/')]))
    test['Time'] = test['Time'].apply(lambda x: x.replace('.000',''))
    test['Time'] = test['Time'].apply(lambda x: ':'.join([i.zfill(2) for i in x.split(':')]))
    test['datetime'] = test.apply(lambda x: datetime.strptime(x.Date +' '+x.Time,'%m/%d/%Y %I:%M:%S %p'),axis=1)
    empty_cols = [col for col in test.columns if test[col].isnull().all()]
    test[empty_cols] = test[empty_cols].fillna(0)

    for i in test.columns[test.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
        test[i].fillna(test[i].mean(),inplace=True)
    def attacked(datetime,datetime_list):
        if datetime in datetime_list:
            return int(1)
        else:
            return int(0)

    test['attack'] = test['datetime'].apply(lambda x: attacked(x,attacks_set))
    coi = ['datetime'] + [i for i in train.columns if i not in ['Date','Time','datetime']]
    coi += ['attack']
    test = test[coi]
    test = test.sort_values('datetime')

    """
    MINMAX norm
    """
    X_test = test.iloc[:,1:-1].to_numpy()
    normalized_test_X = scaler.transform(X_test)
    normalized_test = test.copy()
    normalized_test.iloc[:,1:-1] = normalized_test_X

    """
    Downsample
    """
    filtered_test = normalized_test.iloc[:,1:].groupby(np.arange(len(test.iloc[:,1:]))//10).median()
    max_ftest = normalized_test.iloc[:,1:].groupby(np.arange(len(test.iloc[:,1:]))//10).max()

    final_test = filtered_test.iloc[:-1,:].copy()
    final_test['attack'] = final_test['attack'].round()
    final_test['datetime'] = [test['datetime'][0] + timedelta(seconds=10*i) for i in range(0,len(final_test))]

    ## Renaming
    pat = re.escape('\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\')

    rename_col = [re.sub(pat,'',i) for i in final_test.columns]
    final_test.columns = rename_col

    ## Sort columns
    sort_col = ['datetime'] + [i for i in final_test if i != 'datetime']
    final_test = final_test[sort_col]
    final_test = final_test.reset_index().drop(columns=['datetime','index'])

    final_test = final_test.reset_index().rename(columns={'index':'timestamp'})

    col_names = final_train.columns[1:].tolist()
    col_ids = {col: i for i, col in enumerate(col_names)}
    src_node_to_node = []
    dst_node_to_node = []
    for src in range(len(col_names)):
        for dst in range(len(col_names)):
            if src != dst:
                src_node_to_node.append(src)
                dst_node_to_node.append(dst)
    node_to_node = torch.tensor((src_node_to_node, dst_node_to_node))

    wadi_data = dict(
        train=torch.tensor(final_train.to_numpy()[:, 1:]).float().T,
        test=torch.tensor(final_test.to_numpy()[:, 1:-1]).float().T,
        test_labels=torch.tensor(final_test.to_numpy()[:, -1].astype(int)).float(),
        node_to_node=node_to_node,
        node_indices=col_ids,
        comment="Original WADI dataset from 9 Oct 2017. Processed as in GDN"
    )
    return wadi_data


def read_and_transform_WADI_2017(base_dir):
    normal_data = pd.read_csv(base_dir / "WADI.A1_9 Oct 2017/WADI_14days.csv", skiprows=4)
    prefix_to_remove = "\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\"
    normal_data.columns = normal_data.columns.str.replace(prefix_to_remove, '', regex=False)

    # Start handling missing values
    na_counts = normal_data.isna().sum()
    columns_to_fill = na_counts[na_counts < 100].index
    columns_to_fill_with_zeros = na_counts[na_counts >= 100].index
    normal_data[columns_to_fill] = normal_data[columns_to_fill].fillna(normal_data[columns_to_fill].mean())
    normal_data[columns_to_fill_with_zeros] = normal_data[columns_to_fill_with_zeros].fillna(0)
    # End handling missing values

    col_names = normal_data.columns[3:].tolist()
    col_ids = {col: i for i, col in enumerate(col_names)}
    src_node_to_node = []
    dst_node_to_node = []
    for src in range(len(col_names)):
        for dst in range(len(col_names)):
            if src != dst:
                src_node_to_node.append(src)
                dst_node_to_node.append(dst)
    node_to_node = torch.tensor((src_node_to_node, dst_node_to_node))
    normal_data = torch.tensor(normal_data.iloc[6*60*60:, 3:].to_numpy().T).float()  # remove first 6 hours of data
    # normal_data = torch.tensor(MinMaxScaler().fit_transform(normal_data).clip(0, 1))
    # torch.save(normal_data, base_dir / "WADI_train.pth")

    abnormal_data = pd.read_csv(base_dir / "WADI.A1_9 Oct 2017/WADI_attackdata.csv")
    prefix_to_remove = "\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\"
    abnormal_data.columns = abnormal_data.columns.str.replace(prefix_to_remove, '', regex=False)

    # Start handling of missing values
    abnormal_na_counts = abnormal_data.isna().sum()
    abnormal_columns_to_fill = abnormal_na_counts[abnormal_na_counts < 100].index
    abnormal_columns_to_fill_with_zeros = abnormal_na_counts[abnormal_na_counts >= 100].index
    abnormal_data[abnormal_columns_to_fill] = abnormal_data[abnormal_columns_to_fill].fillna(abnormal_data[abnormal_columns_to_fill].mean())
    abnormal_data[abnormal_columns_to_fill_with_zeros] = abnormal_data[abnormal_columns_to_fill_with_zeros].fillna(0)
    # End handling of missing values

    abnormal_data_labels = pd.read_csv(base_dir / "WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv", header=1)

    abnormal_label = abnormal_data_labels.iloc[:, -1] == -1
    abnormal_label = torch.tensor(abnormal_label.to_numpy().astype(int))[:-2]  # remove last 2 samples
    abnormal_data = torch.tensor(abnormal_data.iloc[:, 3:].to_numpy().T).float()
    # abnormal_data = torch.tensor(MinMaxScaler().fit_transform(abnormal_data).clip(0, 1))

    wadi_data = dict(
        train=normal_data,
        test=abnormal_data,
        test_labels=abnormal_label,
        node_to_node=node_to_node,
        node_indices=col_ids,
        comment="Original WADI dataset from 9 Oct 2017."
    )
    return wadi_data
