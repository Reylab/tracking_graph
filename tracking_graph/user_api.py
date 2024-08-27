from tracking_graph import create_graph, parse_gml,generate_gml
import h5py
import numpy as np
import networkx as nx
from .graphs_tools import merge_splits,count_segments

def load_tg(filepath):
    """
    Loads a tracking graph from a given filepath.

    Parameters:
        filepath (str): The path to the file containing the tracking graph data.

    Returns:
        G (networkx graph): The loaded tracking graph.
        sortings (dict): A dictionary containing the sorting information for each segment and cluster.
    """
    f = h5py.File(filepath, 'r')
    sortings = {}
    for segment_key in f['sortings'].keys():
        segment = f['sortings'][segment_key]
        segment_label = int(segment['label'][()])
        sortings[segment_label] = {}
        for cluster_key in segment.keys():
            if cluster_key == 'label':
                continue
            cluster = segment[cluster_key]
            cluster_label = cluster['label'][()]
            sortings[segment_label][cluster_label] = {}
            sortings[segment_label][cluster_label]['mean_waveform'] = np.array(cluster['mean_waveform'])
            sortings[segment_label][cluster_label]['N'] = int(cluster['N'][()])
    G=parse_gml(f['graph_gml'].asstr()[()])
    return G, sortings

def get_tg_groups(G, mintrack, merge=True):
    """
    Given a graph `G` and a minimum track value `mintrack`, this function run the tracking graph algorithm. 
    
    Parameters:
        G (networkx.Graph or str): The input graph or the path to a GML file containing the graph.
        mintrack (int): The minimum number of tracks a unit must have to be included in the output.
        merge (bool, optional): If True, the function will merge split groups into larger groups. Defaults to True.
    
    Returns:
        tuple: A tuple containing three lists:
            - `groups` (list): A list of groups of nodes in `G` that are present in at least `mintrack` segments.
            - `sG` (networkx.DiGraph): The subgraph of `G` containing only edges with a weight greater than 0.5.
            - `all_groups` (list): A list of all groups of nodes in `G`, including those with less than `mintrack` segments.
    """

    sG = nx.DiGraph(((u, v, e) for u,v,e in G.edges(data=True) if e['weight'] >0.5))
    cG = nx.Graph(((u, v, e) for u,v,e in sG.edges(data=True) if sG.has_edge(v, u)))
    cG.add_nodes_from(G.nodes)
    if merge:
        groups =  merge_splits(list(nx.connected_components(cG)), sG,mintrack)
    else:
        groups =  list(nx.connected_components(cG))
    groups.sort(reverse=True,key=len)
    discarted = sum([list(g) for g in groups if count_segments(g)<mintrack],[])
    groups = [g for g in groups if count_segments(g)>=mintrack]
    return groups, sG, discarted

def run_tg(we_list,outputfile, max_len=2, modelcreator=None):
    """
    This function runs the Tracking_Graph algorithm on a list of waveforms_extractors and saves the output to an HDF5 file.

    Parameters:
        we_list (list): A list of waveforms_extractors to process.
        outputfile (str): The path to the output HDF5 file.
        modelcreator (optional): An optional parameter to specify the model to use to define the edges weights. Defaults to None.
    
    Returns:
        G (networkx.DiGraph): The tracking graph.
    """
    sortings_info = {}
    for i,we in enumerate(we_list):
        sorting = we.sorting
        sortings_info[i] = {}
        for u in sorting.get_unit_ids():
            w_template = we.get_template(u,mode='average')
            nspikes = len(sorting.get_unit_spike_train(unit_id=u, segment_index=0))
            sortings_info[i][u] = {'mean_waveform': w_template, 'N':nspikes}

    G = create_graph(we_list, max_len = max_len,modelcreator=modelcreator)
    g_gml = "\n".join(generate_gml(G))
    if outputfile is not None:
        with h5py.File(outputfile, "w")as hdf_file:
            sortings_group = hdf_file.create_group('sortings')
            for i,(segment, sorting) in enumerate(sortings_info.items()):
                segment_group = sortings_group.create_group('segment_{}'.format(segment))
                segment_group.create_dataset('label',data=segment)
                for cluster_label,u in sorting.items():
                    cluster_group = segment_group.create_group('cluster_{}'.format(cluster_label))
                    cluster_group.create_dataset('mean_waveform', data=u['mean_waveform'])
                    cluster_group.create_dataset('N', data=u['N'])
                    cluster_group.create_dataset('label', data=cluster_label)
            sortings_group = hdf_file.create_dataset('graph_gml', data=g_gml)
    return G