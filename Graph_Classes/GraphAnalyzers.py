import networkx as nx
import matplotlib.pyplot as plt
import random
from networkx.algorithms import bipartite
from collections import deque
import numpy as np


class GraphAnalyzer:
    def __init__(self, graph):
        """
        Initialize the GraphAnalyzer with a graph.
        """
        self.graph = graph

    # Existing methods like plot_degree_distribution, plot_degree_vs_clustering, etc.

    def calculate_average_clustering(self):
        """
        Calculate and return the average clustering coefficient of the graph.
        """
        avg_clustering = nx.average_clustering(self.graph)
        print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
        return avg_clustering

    def plot_degree_centrality_distribution(self):
        """
        Plot the degree centrality distribution of the graph.
        """
        centrality = nx.degree_centrality(self.graph)
        plt.hist(centrality.values(), bins=50, log=True, alpha=0.75, color='orange')
        plt.title("Degree Centrality Distribution")
        plt.xlabel("Centrality")
        plt.ylabel("Frequency (log scale)")
        plt.grid(True)
        plt.show()

    def plot_betweenness_centrality_distribution(self):
        """
        Plot the betweenness centrality distribution of the graph.
        """
        centrality = nx.betweenness_centrality(self.graph)
        plt.hist(centrality.values(), bins=50, log=True, alpha=0.75, color='red')
        plt.title("Betweenness Centrality Distribution")
        plt.xlabel("Centrality")
        plt.ylabel("Frequency (log scale)")
        plt.grid(True)
        plt.show()

    def analyze_connected_components(self):
        """
        Analyze and print the size of connected components in the graph.
        """
        components = [len(c) for c in nx.connected_components(self.graph)]
        plt.hist(components, bins=50, alpha=0.75, color='purple')
        plt.title("Connected Component Size Distribution")
        plt.xlabel("Size of Component")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        print(f"Number of Connected Components: {len(components)}")
        print(f"Sizes of the Largest Components: {sorted(components, reverse=True)[:5]}")

    def calculate_global_efficiency(self):
        """
        Calculate and return the global efficiency of the graph.
        """
        efficiency = nx.global_efficiency(self.graph)
        print(f"Global Efficiency: {efficiency:.4f}")
        return efficiency

    def calculate_assortativity(self):
        """
        Calculate and return the degree assortativity coefficient of the graph.
        """
        assortativity = nx.degree_assortativity_coefficient(self.graph)
        print(f"Degree Assortativity Coefficient: {assortativity:.4f}")
        return assortativity

    def calculate_network_diameter(self):
        """
        Calculate and return the diameter of the graph.
        Only applicable if the graph is connected.
        """
        if nx.is_connected(self.graph):
            diameter = nx.diameter(self.graph)
            print(f"Network Diameter: {diameter}")
            return diameter
        else:
            print("Graph is not connected; diameter cannot be computed.")
            return None

    def plot_closeness_centrality_distribution(self):
        """
        Plot the closeness centrality distribution of the graph.
        """
        centrality = nx.closeness_centrality(self.graph)
        plt.hist(centrality.values(), bins=50, log=True, alpha=0.75, color='cyan')
        plt.title("Closeness Centrality Distribution")
        plt.xlabel("Centrality")
        plt.ylabel("Frequency (log scale)")
        plt.grid(True)
        plt.show()



class BipartiteGraphAnalyzer:
    def __init__(self, graph):
        """
        Initialize the BipartiteGraphAnalyzer with a graph.
        """
        if not nx.is_bipartite(graph):
            raise ValueError("The input graph is not bipartite.")
        self.graph = graph
        self.top_nodes, self.bottom_nodes = bipartite.sets(graph)

    def plot_degree_distribution(self):
        """
        Plot degree distribution for top and bottom node sets.
        """
        top_degrees = [self.graph.degree(n) for n in self.top_nodes]
        bottom_degrees = [self.graph.degree(n) for n in self.bottom_nodes]

        plt.hist(top_degrees, bins=50, alpha=0.6, label="Top Nodes", color="blue")
        plt.hist(bottom_degrees, bins=50, alpha=0.6, label="Bottom Nodes", color="green")
        plt.title("Degree Distribution (Bipartite)")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_bipartite_clustering(self):
        """
        Calculate and return the bipartite clustering coefficient.
        """
        clustering = bipartite.clustering(self.graph)
        avg_clustering = sum(clustering.values()) / len(clustering)
        print(f"Bipartite Clustering Coefficient: {avg_clustering:.4f}")
        return avg_clustering

    def project_to_one_mode(self, project_on="top"):
        """
        Project the bipartite graph to one of its node sets.
        Parameters:
        - project_on: Specify 'top' or 'bottom' for the set to project on.
        Returns:
        - A projected one-mode graph.
        """
        if project_on == "top":
            return bipartite.projected_graph(self.graph, self.top_nodes)
        elif project_on == "bottom":
            return bipartite.projected_graph(self.graph, self.bottom_nodes)
        else:
            raise ValueError("Invalid projection option. Use 'top' or 'bottom'.")

    def plot_degree_correlation(self):
        """
        Plot degree correlation between top and bottom node sets.
        """
        correlations = []
        for node in self.top_nodes:
            neighbor_degrees = [self.graph.degree(n) for n in self.graph.neighbors(node)]
            if neighbor_degrees:
                correlations.append((self.graph.degree(node), sum(neighbor_degrees) / len(neighbor_degrees)))

        if correlations:
            x, y = zip(*correlations)
            plt.scatter(x, y, alpha=0.6, color="orange")
            plt.title("Degree Correlation (Top vs Bottom Nodes)")
            plt.xlabel("Degree of Top Node")
            plt.ylabel("Average Degree of Connected Bottom Nodes")
            plt.grid(True)
            plt.show()
        else:
            print("No degree correlation data available.")

    def calculate_density(self):
        """
        Calculate the density of the bipartite graph.
        """
        density = bipartite.density(self.graph, self.top_nodes)
        print(f"Bipartite Graph Density: {density:.4f}")
        return density

    def analyze_projection(self, projection):
        """
        Analyze the degree distribution and clustering of a projected graph.
        Parameters:
        - projection: A one-mode projection graph.
        """
        degrees = [d for _, d in projection.degree()]
        clustering_coeffs = list(nx.clustering(projection).values())

        plt.hist(degrees, bins=50, alpha=0.6, color="blue", label="Degree Distribution")
        plt.title("Degree Distribution of Projected Graph")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        plt.hist(clustering_coeffs, bins=50, alpha=0.6, color="green", label="Clustering Coefficients")
        plt.title("Clustering Coefficient Distribution (Projected Graph)")
        plt.xlabel("Clustering Coefficient")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    def plot_bipartite_layout(self):
        """
        Plot the bipartite graph using a bipartite layout.
        """
        pos = nx.drawing.layout.bipartite_layout(self.graph, self.top_nodes)
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue", edge_color="gray")
        plt.title("Bipartite Graph Layout")
        plt.show()
