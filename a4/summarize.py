import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
from collections import Counter, defaultdict, deque
import copy
import math
import urllib.request


def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.

    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque

    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.

    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree

    In the doctests below, we first try with max_depth=5, then max_depth=2.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> sorted(node2distances.items())
    [('A', 3), ('B', 2), ('C', 3), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('A', 1), ('B', 1), ('C', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('A', ['B']), ('B', ['D']), ('C', ['B']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> sorted(node2distances.items())
    [('B', 2), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('B', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('B', ['D']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    """
    ###TODO

    #init return value
    node2distances = defaultdict(int)
    node2num_paths = defaultdict(int)
    node2parents = defaultdict(list)

    #add known data to return value
    node2distances[root] = 0
    node2num_paths[root] = 1

    #create a deque
    q = deque()

    #add root to deque
    q.append(root)

    while q:
    	pop = q.popleft()
    	for w in graph.neighbors(pop):
    		if w not in node2distances:
    			node2distances[w] = node2distances[pop] + 1
    			if node2distances[pop] + 1 < max_depth:
    				q.append(w)
    		if node2distances[pop] == node2distances[w] - 1:
    			node2parents[w].append(pop)
    			node2num_paths[w] += 1

    return node2distances, node2num_paths, node2parents


def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    See p 352 From your text:
    https://github.com/iit-cs579/main/blob/master/read/lru-10.pdf
        The third and final step is to calculate for each edge e the sum
        over all nodes Y of the fraction of shortest paths from the root
        X to Y that go through e. This calculation involves computing this
        sum for both nodes and edges, from the bottom. Each node other
        than the root is given a credit of 1, representing the shortest
        path to that node. This credit may be divided among nodes and
        edges above, since there could be several different shortest paths
        to the node. The rules for the calculation are as follows: ...

    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

      Any edges excluded from the results in bfs should also be exluded here.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> result = bottom_up('E', node2distances, node2num_paths, node2parents)
    >>> sorted(result.items())
    [(('A', 'B'), 1.0), (('B', 'C'), 1.0), (('B', 'D'), 3.0), (('D', 'E'), 4.5), (('D', 'G'), 0.5), (('E', 'F'), 1.5), (('F', 'G'), 0.5)]
    """
    ###TODO

    #init node value
    node_dict = defaultdict(int)
    for n in node2distances.keys():
        node_dict[n] = 1
    node_dict[root] = 0

    #create edge dict
    edge_dict = defaultdict(int)

    #descend sort node2distances based on second element to get a node list from far to near
    t = sorted(node2distances.items(), key=lambda d: -d[1])

    #add these nodes to a list
    far2near = []
    for i in t:
        far2near.append(i[0])

    for n in far2near:

        #if node has parents, add its parents to a list
        if n in node2parents:
            parent_list = node2parents[n]

            #sum all the paths
            total_path = 0
            for p in parent_list:
                total_path += node2num_paths[p]

            for p in parent_list:

                #compute the edge credit from node to each one of its parents
                edge_credit = node2num_paths[p] * node_dict[n] / total_path

                #assign new value to parent node
                node_dict[p] += edge_credit

                #add sorted edgea and its credit to edge_dict
                edge_dict[tuple(sorted((n, p)))] += edge_credit

    return dict(edge_dict)

def read_graph(filename):
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Done for you.
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist(filename, delimiter='\t')

def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """
    ###TODO
    nodes = []

    #get a dict of node degree. k is node, v is node degree
    d = graph.degree()

    for k, v in d.items():
    	if (v >= min_degree):
    		nodes.append(k)

    return graph.subgraph(nodes)

def partition_girvan_newman(graph, max_depth):
    """
    Use your approximate_betweenness implementation to partition a graph.
    Unlike in class, here you will not implement this recursively. Instead,
    just remove edges until more than one component is created, then return
    those components.
    That is, compute the approximate betweenness of all edges, and remove
    them until multiple comonents are created.

    You only need to compute the betweenness once.
    If there are ties in edge betweenness, break by edge name (e.g.,
    (('A', 'B'), 1.0) comes before (('B', 'C'), 1.0)).

    Note: the original graph variable should not be modified. Instead,
    make a copy of the original graph prior to removing edges.
    See the Graph.copy method https://networkx.github.io/documentation/development/reference/generated/networkx.Graph.copy.html
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A list of networkx Graph objects, one per partition.

    >>> components = partition_girvan_newman(example_graph(), 5)
    >>> components = sorted(components, key=lambda x: sorted(x.nodes())[0])
    >>> sorted(components[0].nodes())
    ['A', 'B', 'C']
    >>> sorted(components[1].nodes())
    ['D', 'E', 'F', 'G']
    """
    ###TODO

    #create a descending order betweenness 
    b = approximate_betweenness(graph, max_depth)
    sorted_b = sorted(b.items(), key=lambda d: (-d[1], d[0][0], d[0][1]))

    #make a copy of graph
    g = graph.copy()

    #init components list
    components = []
    for item in sorted_b:

    	#when a edge removed, try to form subgraphs 
        g.remove_edge(item[0][0], item[0][1])
        components = [c for c in nx.connected_component_subgraphs(g)]

        #if component has 1 more, then break
        if len(components) > 1: break
    return components

def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.

    You should call the bfs and bottom_up functions defined above for each node
    in the graph, and sum together the results. Be sure to divide by 2 at the
    end to get the final betweenness.

    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A dict mapping edges to betweenness. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

    >>> sorted(approximate_betweenness(example_graph(), 2).items())
    [(('A', 'B'), 2.0), (('A', 'C'), 1.0), (('B', 'C'), 2.0), (('B', 'D'), 6.0), (('D', 'E'), 2.5), (('D', 'F'), 2.0), (('D', 'G'), 2.5), (('E', 'F'), 1.5), (('F', 'G'), 1.5)]
    """
    ###TODO

    #init a dict for summing up
    temp = {}

    #sum each node edge dict credit together
    for n in graph.nodes():
    	node2distances, node2num_paths, node2parents = bfs(graph, n, max_depth)
    	n_edge_dict = bottom_up(n, node2distances, node2num_paths, node2parents)
    	temp = (Counter(n_edge_dict) + Counter(temp)).copy()

    #init a result dict
    result = {}

    #divide the sum by 2
    for k, v in temp.items():
    	result[k] = v / 2

    return result



def number_of_users(filename, output):
	number_of_users = 0
	f = open(filename)
	for line in f:
		number_of_users += 1
	w = open(output, "w")
	w.write('Number of users collected: %d\n' % number_of_users)
	w.close()

def number_of_messages(output):
	w = open(output, "a")
	w.write('\nNumber of messages collected: 100\n')
	w.close()


def number_of_communities(filename, output):
	graph = read_graph(filename)
	subgraph = get_subgraph(graph, 2)
	clusters = partition_girvan_newman(subgraph, 3)
	w = open(output, "a")
	w.write('\nNumber of communities discovered: %d\n' % len(clusters))
	w.close()


def average_number_of_users_per_community(filename, output):
	total = 0
	graph = read_graph(filename)
	subgraph = get_subgraph(graph, 2)
	clusters = partition_girvan_newman(subgraph, 3)
	for i in range(len(clusters)):
		total += clusters[i].order()
	w = open(output, "a")
	w.write('\nAverage number of users per community: %.2f\n' % (float(total) / float(len(clusters))))
	w.close()


def number_of_instance_per_class_found(filename1, filename2, filename3, output):
	n1 = 0
	n2 = 0
	n3 = 0
	for line in open(filename1):
		n1 += 1
	for line in open(filename2):
		n2 += 1
	for line in open(filename3):
		n3 += 1
	w = open(output, "a")
	w.write('\nNumber of instances per class found:\n')
	w.write('Good emotion: %d\n' % (n1 / 2))
	w.write('Neutral emotion: %d\n' % (n2/2))
	w.write('Bad emotion: %d\n' % (n3/2))
	w.close()


def example_from_each_class(filename1, filename2, filename3, output):
    w = open(output, "a")
    w.write("\nOne example from each class:\n")
    f1 = open(filename1,"r")
    w.write("example from Good emotion: %s\n" % str(f1.readline()))
    f1.close()
    f2 = open(filename2,"r")
    w.write("example from Neutral emotion: %s\n" % str(f2.readline()))
    f2.close()
    f3 = open(filename3,"r")
    w.write("example from Bad emotion: %s\n" % str(f3.readline()))
    f3.close()
    w.close()


def main():
	output = open("summary.txt", "w")
	output.close()
	number_of_users('candidates.txt', "summary.txt")
	number_of_messages("summary.txt")
	number_of_communities('output_user.txt', "summary.txt")
	average_number_of_users_per_community('output_user.txt', "summary.txt")
	number_of_instance_per_class_found('output_good_emotion.txt', 'output_neutral_emotion.txt', 'output_bad_emotion.txt', "summary.txt")
	example_from_each_class('output_good_emotion.txt', 'output_neutral_emotion.txt', 'output_bad_emotion.txt', "summary.txt")



if __name__ == '__main__':
    main()
