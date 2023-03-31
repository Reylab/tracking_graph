import networkx as nx
import numpy as np
from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt

#Different colors:
leicolors_list = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.620690, 0.0, 0.0], [0.413793, 0.0, 0.758621],
                  [0.965517, 0.517241, 0.034483], [0.448276, 0.379310, 0.241379], [1.0, 0.103448, 0.724138],
                  [0.545, 0.545, 0.545], [0.586207, 0.827586, 0.310345], [0.965517, 0.620690, 0.862069],
                  [0.620690, 0.758621, 1.]] #silly name, just colors that look different enough
leicolors = lambda x: leicolors_list[int(x) % len(leicolors_list)]

def edges_metrics(model, we,bl_model,bl_spk, std_mult=3):
    """
    Given model, classify the spikes in we into the existing units
    """
    var_mul = std_mult**2
    out = []
    for u in we.sorting.get_unit_ids():
        waveforms = we.get_waveforms(u)
        assign = np.zeros(len(model))
        for i in range(waveforms.shape[0]):
            distances = np.ones(len(model))* np.inf
            for m_i, m_unit_info in enumerate(model.values()):
                d = np.sum((m_unit_info['mean']-waveforms[i,:,:])**2)
                if d <= m_unit_info['sumvar']*var_mul:
                    distances[m_i] = d
            min_m_i = np.argmin(distances)
            if distances[min_m_i] != np.inf:
               assign[min_m_i] = assign[min_m_i] + 1
                
        for m_i,m_unit in enumerate(model.keys()):
            out.append(((int(u),int(bl_spk)),(int(m_unit),int(bl_model)), {"weight": assign[m_i]/waveforms.shape[0]}))
            
    return out

def merge_splits(original, sG,mintrack):
    """
    Given a list of conected elements in original, the graph sG and the number 
    of minimum blocks (mintrack) in which the a unit must be to not be merged.  
    """
    changed = True
    groups = original.copy()
    while changed:
        changed = False
        adj = np.zeros((len(groups),len(groups)))
        for i,nodesi in enumerate(groups[0:-1]):
            for j,nodesj in enumerate(groups[i+1:]):
                for node1, node2 in product(nodesi, nodesj):
                    if sG.has_edge(node1,node2):
                        adj[i,j+1+i] = 1
                    elif  sG.has_edge(node2,node1):
                        adj[j+1+i,i] = 1
        nelements = np.array([len(g) for g in groups])
        outputs = adj.sum(1)
        inputs = adj.sum(0)
        splitsbool = (nelements<mintrack) & (inputs == 0) & (outputs == 1) 
        if any(splitsbool):
            splits = np.where(splitsbool)[0]
            assignto = [np.where(adj[s,:])[0][0] for s in splits]
            maingroups = np.where(np.logical_not(splitsbool))[0]
            new_groups = []
            changed = True
            for gi in maingroups:
                toadd = [groups[splits[mi]] for mi,m in enumerate(assignto) if m==gi]
                for ta in toadd:
                    groups[gi] |= ta
                new_groups.append(groups[gi])
            groups = new_groups
    return groups

def find_pos_genetic(sG, groups,  generations=500, population=500, mutation=0.1 , seed=None):
    blocks_per_group = []
    for nodes in groups:
        blocks_per_group.append(set([n[1] for n in nodes]))
    compatible = nx.Graph()
    compatible.add_nodes_from(range(len(groups)))
    connected = nx.Graph()
    connected.add_nodes_from(range(len(groups)))

    for i,nodesi in enumerate(groups[0:-1]):
        for j,nodesj in enumerate(groups[i+1:]):
            if not blocks_per_group[i].intersection(blocks_per_group[j+1+i]):
                compatible.add_edge(i, j+1+i)
            connections = 0
            for node1, node2 in product(nodesi, nodesj):
                if sG.has_edge(node1,node2) or sG.has_edge(node2,node1) :
                    connections +=1
            connected.add_edge(i, j+1+i, n=connections)
    
    
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    nodes = list(connected.nodes())
    nnodes = len(nodes)

    pos = np.random.permutation(nnodes)
    Nmut = np.ceil(nnodes*mutation).astype(int)
    for gi in range(generations):
        new_pos= mutate_pos(pos, compatible, Nmut,population)
        #calculate metric for each new position
        metrics = np.array([max(p)/len(p) for p in new_pos])
        for u,v,data in connected.edges(data=True):
            metrics += data['n']*abs(new_pos[:,u]-new_pos[:,v])
        pos=new_pos[np.argmin(metrics),:]
    return pos



def mutate_pos(old_pos_vec, compatible, Nmut, population):
    pos = np.tile(old_pos_vec,[population,1])
    for pi in range(population):
        for i in range(Nmut):
            element = np.random.randint(0,old_pos_vec.shape[0])
            mpos = max(pos[pi,:])
            old_pos = pos[pi,element]
            new_pos = np.random.randint(-1, old_pos_vec.shape[0]+1)
            if new_pos>=element:
                new_pos = new_pos +1
            
            if new_pos==-1:
                pos[pi, :] = pos[pi, :] +1
                pos[pi, element] = 0
            elif new_pos == mpos+1:
                pos[pi, element] = mpos+1
            else:
                samepos = np.where(pos[pi,:]==new_pos)[0]
                pcomp = True
                for u in samepos:
                    pcomp = pcomp and compatible.has_edge(u, element)
                if not pcomp:
                    new_pos = new_pos + np.random.randint(0,2)*2-1
                    if new_pos<0:
                        new_pos = 0
                    p2move = pos[pi,:]>=new_pos
                    pos[pi,p2move] = pos[pi,p2move] +1
                pos[pi,element] = new_pos
                    
            if not any(old_pos==pos[pi,:]):
                pos[pi,pos[pi,:]>old_pos] = pos[pi,pos[pi,:]>old_pos] -1
    return pos

def plot_complex_graph(G, sG,cG, groups, gfilename,fancy_pos = True,realgroup=None):
    if realgroup is None:
        realgroup = groups
    nodes_pos = {}
    colors = {}
    maxtime = 0
    if fancy_pos:
        gpos=find_pos_genetic(sG, groups)
        for i,g in enumerate(groups):
            for n in g:
                maybe = (n[1],gpos[i])
                while maybe in nodes_pos.values():
                    maybe = (n[1],maybe[1]+0.7)
                nodes_pos[n] = maybe
                colors[n] = 'k'

        for i,g in enumerate(realgroup):
            for n in g:
                colors[n] = leicolors(i)
                maxtime = max(maxtime, n[1])
    else:
        for i,g in enumerate(groups):
            for n in g:
                nodes_pos[n] = (n[1],n[0])
                colors[n] = i


    edges_short=[(u, v, e) for u,v,e in cG.edges(data=True) if abs(u[1]-v[1])<2]
    edges_long=[(u, v, e) if (u[1]-v[1])<1 else (v, u, e)  for u,v,e in cG.edges(data=True) if  abs(u[1]-v[1])>1]


    edges_short_uni=[(u, v, e) for u,v,e in sG.edges(data=True) if not sG.has_edge(v, u) and abs(u[1]-v[1])<2]

    edges_long_uni_arcn = []
    edges_long_uni_arcp = []
    for u,v,e in cG.edges(data=True):
        if not cG.has_edge(v, u) and abs(v[1]-u[1])>1:
            if (v[1]<u[1]) ^ (nodes_pos[v][1]<=nodes_pos[u][1]):
                edges_long_uni_arcn.append((u, v, e))
            else:
                edges_long_uni_arcp.append((u, v, e))
    fig, ax =plt.subplots(figsize=(18,6))
    nx.draw_networkx_nodes(G, nodes_pos, node_color=[ colors[nd] for nd in G.nodes],edgecolors='k')
    nx.draw_networkx_edges(G, nodes_pos, edgelist=edges_short,width=1.5, edge_color='blue',arrowstyle='-')
    nx.draw_networkx_edges(G, nodes_pos, edgelist=edges_long, width=1, edge_color='blue',connectionstyle="arc3,rad=0.35",arrows=True,arrowstyle='-')
    nx.draw_networkx_edges(G, nodes_pos, edgelist=edges_short_uni,width=1, edge_color='r')
    nx.draw_networkx_edges(G, nodes_pos, edgelist=edges_long_uni_arcn, width=1, edge_color='orange',connectionstyle="arc3,rad=-0.3")
    nx.draw_networkx_edges(G, nodes_pos, edgelist=edges_long_uni_arcp, width=1, edge_color='orange',connectionstyle="arc3,rad=0.3")
    
    ax.set_axis_on()
    ax.tick_params(bottom=True, labelbottom=True)

    plt.xticks(range(0,maxtime+1),range(1,maxtime+2))
    plt.xlim([-0.5,maxtime+0.5])
    plt.tight_layout()
    fig.savefig(gfilename,dpi=600)
    plt.close(fig)

