import networkx as nx
from typing import Dict, Any
from collections import Counter


class StructureEvaluator:
    """Evaluate knowledge graph structure using graph metrics"""

    def __init__(self):
        pass

    def evaluate(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Evaluate graph structure and return metrics

        Args:
            graph: NetworkX DiGraph with nodes containing 'text' and 'embedding'

        Returns:
            Dict containing all structure evaluation metrics
        """
        if graph.number_of_nodes() == 0:
            return self._empty_metrics()

        metrics = {
            "basic_stats": self._compute_basic_stats(graph),
            "connectivity": self._compute_connectivity(graph),
            "centrality": self._compute_centrality(graph),
            "path_metrics": self._compute_path_metrics(graph),
            "relation_metrics": self._compute_relation_metrics(graph)
        }

        return metrics

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics for empty graph"""
        return {
            "basic_stats": {
                "num_nodes": 0,
                "num_edges": 0,
                "density": 0.0,
                "num_relation_types": 0,
                "avg_degree": 0.0,
                "num_isolated_nodes": 0
            },
            "connectivity": {
                "is_connected": False,
                "num_connected_components": 0,
                "largest_component_size": 0,
                "largest_component_ratio": 0.0
            },
            "centrality": {
                "avg_clustering_coefficient": 0.0,
                "avg_degree_centrality": 0.0,
                "max_degree_centrality": 0.0
            },
            "path_metrics": {
                "avg_shortest_path_length": 0.0,
                "diameter": 0
            },
            "relation_metrics": {
                "num_relation_types": 0,
                "relation_distribution": {},
                "top_relations": []
            }
        }

    def _compute_basic_stats(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Compute basic graph statistics"""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        # Density
        density = nx.density(graph)

        # Average degree (for directed graph, use total degree)
        degrees = [d for n, d in graph.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

        # Isolated nodes (degree = 0)
        num_isolated = sum(1 for n, d in graph.degree() if d == 0)

        # Relation types
        relations = set()
        for u, v, data in graph.edges(data=True):
            if 'relation' in data:
                relations.add(data['relation'])
        num_relation_types = len(relations)

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": round(density, 6),
            "num_relation_types": num_relation_types,
            "avg_degree": round(avg_degree, 2),
            "num_isolated_nodes": num_isolated
        }

    def _compute_connectivity(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Compute connectivity metrics"""
        # Convert to undirected for connectivity analysis
        undirected = graph.to_undirected()

        is_connected = nx.is_connected(undirected)
        num_components = nx.number_connected_components(undirected)

        # Largest component
        if num_components > 0:
            largest_cc = max(nx.connected_components(undirected), key=len)
            largest_size = len(largest_cc)
            largest_ratio = largest_size / graph.number_of_nodes()
        else:
            largest_size = 0
            largest_ratio = 0.0

        return {
            "is_connected": is_connected,
            "num_connected_components": num_components,
            "largest_component_size": largest_size,
            "largest_component_ratio": round(largest_ratio, 4)
        }

    def _compute_centrality(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Compute centrality metrics"""
        # Clustering coefficient (use undirected version)
        undirected = graph.to_undirected()
        clustering_coeffs = nx.clustering(undirected)
        avg_clustering = sum(clustering_coeffs.values()) / len(clustering_coeffs) if clustering_coeffs else 0.0

        # Degree centrality
        degree_centrality = nx.degree_centrality(graph)
        avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality) if degree_centrality else 0.0
        max_degree_centrality = max(degree_centrality.values()) if degree_centrality else 0.0

        return {
            "avg_clustering_coefficient": round(avg_clustering, 4),
            "avg_degree_centrality": round(avg_degree_centrality, 6),
            "max_degree_centrality": round(max_degree_centrality, 4)
        }

    def _compute_path_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Compute path-related metrics

        For large graphs (>5000 nodes), skip expensive computations to avoid long runtime
        """
        # Use undirected graph for path calculations
        undirected = graph.to_undirected()

        # Skip expensive computations for large graphs
        LARGE_GRAPH_THRESHOLD = 5000
        if graph.number_of_nodes() > LARGE_GRAPH_THRESHOLD:
            print(f"  Skipping path metrics for large graph ({graph.number_of_nodes()} nodes > {LARGE_GRAPH_THRESHOLD} threshold)")
            return {
                "avg_shortest_path_length": -1.0,  # -1 indicates skipped
                "diameter": -1,  # -1 indicates skipped
                "note": "Skipped for large graph (would take too long)"
            }

        # Only compute for largest connected component
        if nx.is_connected(undirected):
            avg_path_length = nx.average_shortest_path_length(undirected)
            diameter = nx.diameter(undirected)
        else:
            # Get largest component
            largest_cc = max(nx.connected_components(undirected), key=len)
            subgraph = undirected.subgraph(largest_cc)

            if len(largest_cc) > 1:
                avg_path_length = nx.average_shortest_path_length(subgraph)
                diameter = nx.diameter(subgraph)
            else:
                avg_path_length = 0.0
                diameter = 0

        return {
            "avg_shortest_path_length": round(avg_path_length, 2),
            "diameter": diameter
        }

    def _compute_relation_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Compute relation-related metrics"""
        # Collect all relations
        relations = []
        for u, v, data in graph.edges(data=True):
            if 'relation' in data:
                relations.append(data['relation'])

        if not relations:
            return {
                "num_relation_types": 0,
                "relation_distribution": {},
                "top_relations": []
            }

        # Count relations
        relation_counter = Counter(relations)
        num_relation_types = len(relation_counter)

        # Get top 5 relations
        top_5 = relation_counter.most_common(5)
        relation_distribution = {rel: count for rel, count in top_5}
        top_relations = [rel for rel, count in top_5]

        return {
            "num_relation_types": num_relation_types,
            "relation_distribution": relation_distribution,
            "top_relations": top_relations
        }


# Usage example
if __name__ == "__main__":
    from RAGCore.Graph.GraphSave import GraphSaver

    dataset_name = "hotpotqa"

    # Load graph
    print("Loading graph...")
    graph = GraphSaver.load(dataset_name)

    # Evaluate structure
    print("\nEvaluating graph structure...")
    evaluator = StructureEvaluator()
    metrics = evaluator.evaluate(graph)

    # Print results
    print("\n=== Structure Evaluation Results ===")
    print(f"\nBasic Stats:")
    for key, value in metrics["basic_stats"].items():
        print(f"  {key}: {value}")

    print(f"\nConnectivity:")
    for key, value in metrics["connectivity"].items():
        print(f"  {key}: {value}")

    print(f"\nCentrality:")
    for key, value in metrics["centrality"].items():
        print(f"  {key}: {value}")

    print(f"\nPath Metrics:")
    for key, value in metrics["path_metrics"].items():
        print(f"  {key}: {value}")

    print(f"\nRelation Metrics:")
    print(f"  num_relation_types: {metrics['relation_metrics']['num_relation_types']}")
    print(f"  top_relations: {metrics['relation_metrics']['top_relations']}")
    print(f"  relation_distribution: {metrics['relation_metrics']['relation_distribution']}")
