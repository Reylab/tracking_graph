# %%
import networkx as nx
import numpy as np
from itertools import product
import json
from .linking_models import EuclideanClassifier
from collections import namedtuple
tg_node_cl_parent = namedtuple('tg_node', ['segment', 'unit'])
cluster_compare_type = namedtuple('cluster_compare_type', ['Cluster_A', 'Cluster_B'])

class tg_node(tg_node_cl_parent):
    def __str__(self):
        return f"(unit:{self.unit}, segment:{self.segment})"
    @classmethod
    def from_string(cls, string):
        # Extract the values from the string
        prefix = "(unit:"
        suffix = ")"
        if string.startswith(prefix) and string.endswith(suffix):
            body = string[len(prefix):-len(suffix)]
            parts = body.split(", segment:")
            unit = int(parts[0])
            segment = int(parts[1])
            return cls(segment=segment, unit=unit)
        else:
            raise ValueError("String does not match the expected format")




def node_to_string(node):
    return json.dumps(node._asdict())

def string_to_node(text):
    pp = json.loads(text)
    return tg_node(**pp)

def generate_gml(sG):
    return nx.generate_gml(sG, node_to_string)

def write_to_gml(sG, filaname):
    nx.write_gml(sG, filaname, node_to_string)

def parse_gml(s):
    return nx.parse_gml(s,destringizer =string_to_node)

def read_to_gml(filaname):
    return nx.read_gml(filaname,destringizer =string_to_node)


def err_function(x):
    return 1 - 2 * abs(x - 0.5)

vectorized_err_function = np.vectorize(err_function)

def err_connection_function(x,y):
    return 1 - 2 * max(abs(x - 0.5),abs(y - 0.5))

# Apply the function to each element of the matrices
vectorized_err_function = np.vectorize(err_function)
vectorized_err_connection_function = np.vectorize(err_connection_function)

def compute_connection_quality(G,thr=0.5):
    nodelist = list(G.nodes())
    nodelist = sorted(nodelist, key=lambda x: (x.segment, x.unit))
    adj_matrix_forward = nx.to_numpy_array(G, nodelist=nodelist, weight='weight')
    matrix = vectorized_err_connection_function(adj_matrix_forward,adj_matrix_forward.T)
    return matrix, nodelist

def compute_model_quality(G,thr=0.5):
    nodelist = list(G.nodes())
    nodelist = sorted(nodelist, key=lambda x: (x.segment, x.unit))
    adj_matrix_forward = nx.to_numpy_array(G, nodelist=nodelist, weight='weight')

    matrix = vectorized_err_function(adj_matrix_forward)
    return matrix, nodelist

def create_graph(we_list, max_len = 2, modelcreator=None):
    """
    Creates a graph based on the given list of waveform extractors.

    Parameters:
        we_list : A list of waveform extractors.
        modelcreator (Callable, optional): A function that creates a model for fitting the spikes. Defaults to None, in which case EuclideanClassifier.creator(std_mult=3) will be used.

    Returns:
        nx.DiGraph: The created full graph
    """

    if modelcreator is None:
        modelcreator = EuclideanClassifier.creator(std_mult=3)
    nsortings = len(we_list)
    models = []
    for we in we_list:
        model = modelcreator()
        model.fit(we)
        models.append(model)
    G = nx.DiGraph()
    for i in range(nsortings):
        for j in range(i+1,min(i+1+max_len,nsortings)):
            G.add_edges_from( edges_metrics_from_model(models[i], we_list[j], i,j))
            G.add_edges_from( edges_metrics_from_model(models[j], we_list[i], j,i))
    return G

def edges_metrics_from_model(model, we,bl_model,bl_spk):
    """
    Given model, classify the spikes in we into the existing units
    """
    out = []
    for u in we.sorting.get_unit_ids():
        waveforms = we.get_waveforms(u)
        units_rates  = model.get_model_assignation_rates(waveforms) 
        for m_i,m_unit in enumerate(model.get_unit_ids()):
            out.append((tg_node(unit=int(u),segment=int(bl_spk)),
                        tg_node(unit=int(m_unit),segment=int(bl_model)),
                         {"weight": units_rates[m_i]}))
    return out

def count_segments(g):
    return len(np.unique([n.segment for n in g]))

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
        n_blocks = np.array([count_segments(g) for g in groups])
        outputs = adj.sum(1)
        inputs = adj.sum(0)
        splitsbool = (n_blocks<mintrack) & (inputs == 0) & (outputs == 1) 
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


def find_pos_genetic(sG, groups,  generations=300, population=250, 
                     mutation=0.3, survivors=10, seed=0):
    """
    Find the optimal position for each node in the graph using a genetic algorithm.

    Parameters:
        sG (nx.Graph): The graph to find the positions for.
        groups (List[Set[Node]]): The groups of nodes in the graph.
        generations (int, optional): The number of generations to run the genetic algorithm for. Defaults to 300.
        population (int, optional): The number of individuals in each generation. Defaults to 250.
        mutation (float, optional): The probability of mutation for each individual. Defaults to 0.3.
        survivors (int, optional): The number of individuals to keep in each generation. Defaults to 10.
        seed (int, optional): The random seed for the genetic algorithm. Defaults to 0.

    Returns:
        np.ndarray: The optimal position for each node in the graph.

    Raises:
        AssertionError: If the population is not a multiple of the survivors.

    """

    assert population%survivors==0,'survivors must be a divisor of population'
    rng = np.random.default_rng(seed=seed)
    blocks_per_group = []
    for nodes in groups:
        blocks_per_group.append(np.array([n.segment for n in nodes]))
    non_compatible = []
    connected = nx.Graph()
    connected.add_nodes_from(range(len(groups)))

    for i,nodesi in enumerate(groups[0:-1]):
        for j,nodesj in enumerate(groups[i+1:]):
            if np.min(np.abs(np.subtract.outer(blocks_per_group[i], blocks_per_group[j+1+i])))<=1:
            #not blocks_per_group[i].intersection(blocks_per_group[j+1+i]):
                non_compatible.append((i, j+1+i))
                non_compatible.append((j+1+i, i))
            connections = 0
            for node1, node2 in product(nodesi, nodesj):
                if sG.has_edge(node1,node2) or sG.has_edge(node2,node1) :
                    connections +=1
            connected.add_edge(i, j+1+i, n=connections)
    
    nodes = list(connected.nodes())
    nnodes = len(nodes)
    non_compatible_set = frozenset(non_compatible)    

    pos = rng.integers(0, nnodes, size=[survivors,nnodes])
    for pi in range(survivors):
        _,pos[pi,:] = np.unique(pos[pi,:] ,return_inverse=True)
        for element in np.arange(nnodes):
            samepos = np.where(pos[pi,:]==pos[pi,element])[0]
            for u in samepos:
                if (u, element) in non_compatible_set:
                    current_pos = pos[pi,element]
                    p2move = pos[pi,:]>=current_pos
                    pos[pi,p2move] = pos[pi,p2move] +1
                    pos[pi,element] = current_pos
                    break
        _,pos[pi,:] = np.unique(pos[pi,:] ,return_inverse=True)


    Nmut_vec = np.ceil(np.linspace(nnodes*mutation,1,int(generations))).astype(int)
    for mn in Nmut_vec:
        new_pos= mutate_pos(pos, non_compatible_set, mn,population,survivors,rng)
        #calculate metric for each new position
        metrics = np.array([max(p) for p in new_pos])/new_pos.shape[1]
        for u,v,data in connected.edges(data=True):
            metrics += data['n']*(new_pos[:,u]-new_pos[:,v])**2
        pos = new_pos[np.argsort(metrics)[:survivors],:]
        #pos=new_pos[np.argmin(metrics),:]
    return new_pos[np.argmin(metrics),:]

def mutate_pos(old_pos_vec, non_compatible, Nmut, population,survivors,rng):
    """
    Mutates the position old_pos_vec into a valid solution of elements.

    Parameters:
        old_pos_vec (numpy.ndarray): The array representing the current positions.
        non_compatible (set): The set of edges representing the compatibility between nodes.
        Nmut (int): The number of mutations to perform.
        population (int): The total population size.
        survivors (int): The number of surviving individuals.
        rng (numpy.random.Generator): The random number generator.

    Returns:
        numpy.ndarray: The mutated position array.

    """ 
    pos = np.tile(old_pos_vec,[int(population/survivors),1])
    n_elements = old_pos_vec.shape[1]
    for pi in range(survivors,population):
        for _ in range(Nmut):
            element = rng.integers(0,n_elements)
            mpos = np.max(pos[pi,:])
            old_pos = pos[pi,element]
            new_pos = rng.integers(-1, mpos+1)
            if new_pos == old_pos:
               continue
            if new_pos==-1:
                pos[pi, :] = pos[pi, :] +1
                pos[pi, element] = 0
            elif new_pos == mpos+1:
                pos[pi, element] = mpos+1
            else:
                samepos = np.where(pos[pi,:]==new_pos)[0]
                for u in samepos:
                    if (u, element) in non_compatible:
                        new_pos = new_pos + rng.integers(0,2)*2-1
                        new_pos = max(new_pos,0)
                        p2move = pos[pi,:] >= new_pos
                        pos[pi,p2move] = pos[pi,p2move] +1
                        break
                pos[pi,element] = new_pos
                    
            if np.all(old_pos != pos[pi,:]):
                to_reduce = pos[pi,:]>old_pos
                pos[pi,to_reduce] = pos[pi,to_reduce] -1
    return pos
