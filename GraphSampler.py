import networkx as nx
import random
from collections import deque

class GraphSampler:
    def __init__(self, G):
        """
        Initialize the GraphSampler with a graph G.
        """
        self.G = G
    
    def metropolis_hastings_sampling(self, start_node, num_samples):
        """
        Perform Metropolis-Hastings Sampling on the graph.
        """
        sampled_graph = nx.Graph()
        sampled_graph.add_node(start_node)
        
        current_node = start_node
        
        while sampled_graph.number_of_nodes() < num_samples:
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                print("No neighbors to move to. Sampling stopped.")
                break
            
            next_node = random.choice(neighbors)
            current_degree = self.G.degree(current_node)
            next_degree = self.G.degree(next_node)
            
            acceptance_ratio = min(1, next_degree / current_degree)
            
            if random.random() < acceptance_ratio:
                sampled_graph.add_node(next_node)
                sampled_graph.add_edge(current_node, next_node)
                current_node = next_node
            else:
                sampled_graph.add_node(current_node)
        
        return sampled_graph

    def random_walk_sampling(self, start_node, num_samples):
        """
        Perform Random Walk Sampling on the graph.
        """
        sampled_graph = nx.Graph()
        sampled_graph.add_node(start_node)
        
        current_node = start_node
        visited = set()
        visited.add(current_node)
        
        while sampled_graph.number_of_nodes() < num_samples:
            neighbors = list(self.G.neighbors(current_node))
            if not neighbors:
                print("No neighbors to move to. Sampling stopped.")
                break
            
            next_node = random.choice(neighbors)
            
            if next_node not in visited:
                sampled_graph.add_node(next_node)
                sampled_graph.add_edge(current_node, next_node)
                visited.add(next_node)
            
            current_node = next_node
        
        return sampled_graph

    def snowball_sampling(self, initial_nodes, sample_size, expand_depth=1):
        """
        Perform snowball sampling on the graph starting from initial nodes.
        """
        sampled_nodes = set(initial_nodes)
        to_expand = set(initial_nodes)
        
        for _ in range(expand_depth):
            next_expand = set()
            
            for node in to_expand:
                neighbors = list(self.G.neighbors(node))
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if len(sampled_nodes) < sample_size:
                        if neighbor not in sampled_nodes:
                            sampled_nodes.add(neighbor)
                            next_expand.add(neighbor)
                    else:
                        break
            
            to_expand = next_expand
            
            if len(sampled_nodes) >= sample_size:
                break
        
        return list(sampled_nodes)

    def depth_first_sampling(self, initial_node, sample_size):
        """
        Perform depth-first sampling on the graph starting from an initial node.
        """
        sampled_nodes = set()
        stack = [initial_node]
        
        while stack and len(sampled_nodes) < sample_size:
            node = stack.pop()
            
            if node not in sampled_nodes:
                sampled_nodes.add(node)
                neighbors = list(self.G.neighbors(node))
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if len(sampled_nodes) < sample_size:
                        stack.append(neighbor)
        
        return list(sampled_nodes)

    def breadth_first_sampling(self, initial_nodes, sample_size):
        """
        Perform breadth-first sampling on the graph starting from initial nodes.
        """
        sampled_nodes = set(initial_nodes)
        queue = deque(initial_nodes)
        
        while queue and len(sampled_nodes) < sample_size:
            node = queue.popleft()
            neighbors = list(self.G.neighbors(node))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in sampled_nodes:
                    sampled_nodes.add(neighbor)
                    queue.append(neighbor)
                    
                    if len(sampled_nodes) >= sample_size:
                        break
        
        return list(sampled_nodes)

    def random_edge_sampling(self, sample_size):
        """
        Perform random edge sampling on the graph.
        """
        edges = list(self.G.edges())
        random.shuffle(edges)
        sampled_edges = edges[:sample_size]
        return sampled_edges

    def induced_edge_sampling(self, sample_size):
        """
        Perform induced edge sampling on the graph.
        """
        sampled_edges = []
        nodes = list(self.G.nodes())
        random.shuffle(nodes)
        
        for node in nodes:
            neighbors = list(self.G.neighbors(node))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if len(sampled_edges) < sample_size:
                    sampled_edges.append((node, neighbor))
                else:
                    return sampled_edges
        
        return sampled_edges

    def random_node_sampling(self, sample_size):
        """
        Perform random node sampling on the graph.
        """
        sampled_nodes = random.sample(list(self.G.nodes()), sample_size)
        return sampled_nodes

    def degree_based_node_sampling(self, sample_size):
        """
        Perform degree-based node sampling on the graph.
        """
        nodes_sorted_by_degree = sorted(self.G.nodes(), key=lambda x: self.G.degree(x), reverse=True)
        sampled_nodes = nodes_sorted_by_degree[:sample_size]
        return sampled_nodes
