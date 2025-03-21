import networkx as nx
import pandas as pd


def subgraph(G, start_word, depth=1):
    """Genera y visualiza un subgrafo a partir de una palabra con profundidad dada."""
    if start_word not in G:
        print(f"La palabra '{start_word}' no está en el grafo.")
        return

    # Obtener los nodos alcanzables en el nivel de profundidad especificado
    edges = list(nx.bfs_edges(G, start_word, depth_limit=depth))
    nodes = {start_word} | {v for u, v in edges}

    # Crear el subgrafo
    return G.subgraph(nodes)


def subgraph_df(G, start_word, depth=2):
    """Genera y visualiza un subgrafo a partir de una palabra con profundidad dada."""
    if start_word not in G:
        print(f"La palabra '{start_word}' no está en el grafo.")
        return

    # Obtener los nodos alcanzables en el nivel de profundidad especificado
    edges = list(nx.bfs_edges(G, start_word, depth_limit=depth))
    nodes = [{
        'id': start_word,
        'label': start_word,
        'parent': None,
        'count': None
    }]

    for u, v in edges:
        nodes.append({
            'id': v,
            'label': v,
            'parent': u,
            'count': G[u][v]['weight']
        })

    return pd.DataFrame(nodes)
