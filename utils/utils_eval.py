"""
utility functions for result visualization and evaluation
"""
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def extract_indices(q, include_diag=False):
    """ helper function for determine the indices """
    valid_loc = np.ones([q, q]) if include_diag else np.ones([q, q]) - np.eye(q)
    return  np.where(valid_loc)


def normalize_and_reshape(graph, q, include_diag=False, normalize=True):

    if normalize:
        graph = graph/np.max(np.abs(graph))
    
    graph_new = np.zeros((q,q))
    indices = extract_indices(q, include_diag=include_diag)
    graph_new[indices[0],indices[1]] = graph
    
    return graph_new


def get_auc(truth, est, include_diag=False):
    if not include_diag:
        truth = truth.copy()
        for i in range(truth.shape[-1]):
            truth[i,i] = 0
            
    truth_binary = 1*(truth!=0).ravel()
    auroc = metrics.roc_auc_score(truth_binary, np.abs(est).ravel())
    auprc = metrics.average_precision_score(truth_binary, np.abs(est).ravel())

    return {'auroc':auroc, 'auprc':auprc}


def gather_metrics(true_graph, est_graph, include_diag=False, threshold=None):
    
    indices = extract_indices(true_graph.shape[-1], include_diag=include_diag)
    y_true = 1 * (np.abs(true_graph[indices[0],indices[1]]) != 0)
    y_pred = est_graph[indices[0],indices[1]]
    if threshold:
        y_pred = 1 * (np.abs(y_pred) > threshold)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metric_suite = {}
    
    metric_suite['specificity'] = tn/(tn+fp)
    metric_suite['sensitivity'] = tp/(tp+fn)
    metric_suite['precision']= tp/(tp+fp)
    metric_suite['recall']= tp/(tp+fn)

    metric_suite['f1score'] = 2*tp/(2*tp+fp+fn)
    metric_suite['mcc'] = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    metric_suite['acc'] = (tp+tn)/(tp+tn+fp+fn)
    metric_suite['ba'] = (metric_suite['specificity']+metric_suite['sensitivity'])/2
    
    return metric_suite


def get_heatmap(mtx,
            axs_to_plot,
            rm_diag=False,
            threshold=None,
            threshold_by_quantile=False,
            vmin=None, vmax=None,
            alpha=1, labels=None, annot=False, color_palette='seismic'):

    cmap=sns.color_palette(color_palette, as_cmap=True)

    mtx = mtx.copy()
    if rm_diag:
        for i in range(mtx.shape[0]):
            mtx[i,i] = 0

    vmin = vmin or -1*np.max(np.abs(mtx))
    vmax = vmax or np.max(np.abs(mtx))

    if threshold is not None:
        if threshold_by_quantile:
            threshold = np.quantile(np.abs(mtx),threshold)
        mtx = mtx * 1 * (np.abs(mtx) > threshold)

    if labels is not None:
        mtx = pd.DataFrame(data=mtx, columns=labels, index=labels)

    g = sns.heatmap(mtx, linewidth=0.1, vmin=vmin, vmax=vmax, alpha=alpha,
            cmap=cmap, cbar=False, annot=annot, fmt=".2f", annot_kws={"fontsize":9},ax=axs_to_plot)
    
    g.set_yticklabels(g.get_yticklabels(), size=7, rotation=0)
    g.set_xticklabels(g.get_xticklabels(), size=7, rotation=0)
    for _, spine in axs_to_plot.spines.items():
        spine.set_visible(True)

    return


def get_heatmap_binary(mtx, axs_to_plot, threshold=0.01, rm_diag=False, alpha=1, labels=None, annot=False, color_palette='binary'):
    
    cmap = sns.color_palette(color_palette, as_cmap=True)
    bounds = [0.,threshold,1.]
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    
    mtx = np.abs(mtx)

    if rm_diag:        
        for i in range(mtx.shape[0]):
            mtx[i,i] = 0
    
    g = sns.heatmap(mtx, linewidth=0.1, cmap=cmap, norm=norm, cbar=False,
        annot=annot, fmt=".2f", annot_kws={"fontsize":9}, ax=axs_to_plot, alpha=alpha)
    g.set_yticklabels(g.get_yticklabels(), size=7, rotation=0)
    g.set_xticklabels(g.get_xticklabels(), size=7, rotation=0)
    for _, spine in axs_to_plot.spines.items():
        spine.set_visible(True)
    
    return


def get_network_metrics(G, undirected=True, verbose=False, threshold=None):
    """
    G: either a nx.Graph object or a 2D np.array
    """
    if isinstance(G, np.ndarray):
        if threshold:
            G = G * 1 * (np.abs(G) > threshold)
        graph_type = nx.DiGraph if not undirected else nx.Graph
        G = nx.from_numpy_array(G, create_using=graph_type)
    
    p = G.number_of_nodes()
    
    network_metrics = {}
    network_metrics['density'] = nx.density(G)
    network_metrics['betweeness'] = sum(nx.betweenness_centrality(G).values())/p
    network_metrics['transitivity'] = nx.transitivity(G)
    network_metrics['average_clustering'] = nx.average_clustering(G)
    #network_metrics['average_node_connectivity'] = nx.average_node_connectivity(G)
    
    try:
        network_metrics['avgPathLength'] = nx.average_shortest_path_length(G)
        network_metrics['diameter'] = nx.diameter(G)
    except Exception as e:
        network_metrics['avgPathLength'] = np.nan
        network_metrics['diameter'] = np.nan
        if verbose:
            print(f'WARNING: exception occurred; {str(e)}')
    
    return network_metrics

def get_network_metrics_snapshots(Gs, undirected=True, threshold=None):
    metrics_over_time = []
    for t in range(Gs.shape[0]):
        nx_metrics = get_network_metrics(Gs[t,:,:], undirected=undirected, threshold=threshold)
        metrics_over_time.append({'t':t,**nx_metrics})
    
    return metrics_over_time
