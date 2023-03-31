from graph_tools import edges_metrics, merge_splits
import networkx as nx
import numpy as np
# a list of WaveformExtractors is required: we_list

mintrack = 2

#Then a graph is created
template_info = []
for i, we in enumerate(we_list):
    sorting = we.sorting
    units_info = {}
    for u in sorting.get_unit_ids():
        w_std = np.sum((we.get_template(u,mode='std')**2))
        w_template = we.get_template(u,mode='average')
        units_info[u] = {'mean':w_template, 'sumvar': w_std}
    template_info.append(units_info)
G = nx.DiGraph()
for i in range(len(template_info)):
    for j in range(i+1,min(i+3,len(template_info))):
        G.add_edges_from( edges_metrics(template_info[i], we_list[j], i,j))
        G.add_edges_from( edges_metrics(template_info[j], we_list[i], j,i))



sG = nx.DiGraph(((u, v, e) for u,v,e in G.edges(data=True) if e['weight'] >0.5))
cG = nx.Graph(((u, v, e) for u,v,e in sG.edges(data=True) if sG.has_edge(v, u)))
cG.add_nodes_from(G.nodes)

groups =  merge_splits(list(nx.connected_components(cG)), sG,mintrack)
groups.sort(reverse=True,key=len)
discarted = sum([list(g) for g in groups if len(g)<mintrack],[])
groups = [g for g in groups if len(g)>=mintrack]

#Each element of groups is a list of elements of the type: (#unit,#session) that correspond to the same final unit