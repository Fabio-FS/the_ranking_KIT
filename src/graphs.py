import igraph as ig
import matplotlib.pyplot as plt
import numpy as np







def build_graph(info):
    if "Graph" in info:
        g_type = info["Graph"].get("type", "NULL")
        if g_type == "NULL":
            G = ig.Graph.Erdos_Renyi(n=100, p=0.1, directed=False, loops=False)
        elif(g_type == "ER"):
            n = info["Graph"].get("n", 100)
            p = info["Graph"].get("p", 0.1)
            G = ig.Graph.Erdos_Renyi(n=n, p=p, directed=False, loops=False)
        elif(g_type == "BA"):
            n = info["Graph"].get("n", 100)
            m = info["Graph"].get("m", 2)
            G = ig.Graph.Barabasi(n=n, m=m, directed=False)
        elif(g_type == "WS"):
            n = info["Graph"].get("n", 100)
            k = info["Graph"].get("k", 4)
            p = info["Graph"].get("p", 0.1)
            G = ig.Graph.Watts_Strogatz(dim=1, size=n, nei=k, p=p)
        else:
            raise ValueError(f"Unknown graph type: {g_type}")
    else:
        G = ig.Graph.Erdos_Renyi(n=100, p=0.1, directed=False, loops=False)

    n_users = G.vcount()
    history_size = info.get('post_history', 50)
    all_times = np.arange(history_size)

    G.vs['neighbors'] = [np.array(G.neighbors(i)) for i in range(n_users)]

    neighbor_matrix = np.zeros((n_users, n_users), dtype=bool)
    for i in range(n_users):
        if len(G.vs[i]['neighbors']) > 0:
            neighbor_matrix[i, G.vs[i]['neighbors']] = True
    G['neighbor_matrix'] = neighbor_matrix

    # Precompute per-user neighbor post index arrays (used by all rankers)
    neighbor_authors = []
    neighbor_times = []
    for i in range(n_users):
        neighbors = G.vs[i]['neighbors']
        neighbor_authors.append(np.repeat(neighbors, history_size))
        neighbor_times.append(np.tile(all_times, len(neighbors)))

    G['neighbor_authors'] = neighbor_authors
    G['neighbor_times'] = neighbor_times
    G['edges'] = np.array(G.get_edgelist())
    return G