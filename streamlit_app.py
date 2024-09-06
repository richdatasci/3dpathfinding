import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import heapq
import random
import numpy as np

# Function to generate the graph with compartments and eyeplates
def generate_graph():
    G = nx.Graph()
    
    # Create four large nodes representing compartments below the eyeplates (z = -10)
    bottom_compartments = {'Compartment A': (0, 0, -10), 
                           'Compartment B': (10, 0, -10), 
                           'Compartment C': (0, 10, -10), 
                           'Compartment D': (10, 10, -10)}
    
    # Create four large nodes representing compartments above the eyeplates (z = 10)
    top_compartments = {'Compartment E': (0, 0, 10),
                        'Compartment F': (10, 0, 10),
                        'Compartment G': (0, 10, 10),
                        'Compartment H': (10, 10, 10)}
    
    # Add large compartment nodes (bottom and top compartments)
    for compartment, position in {**bottom_compartments, **top_compartments}.items():
        G.add_node(compartment, pos=position, size=20, group='compartment')

    # Create smaller nodes representing eyeplates (z = 0)
    eyeplates = {}
    for i in range(1, 21):
        x, y, z = random.uniform(0, 10), random.uniform(0, 10), 0  # Z-axis at 0 for eyeplates
        eyeplates[f'Eyeplate {i}'] = (x, y, z)

    # Add eyeplate nodes
    for eyeplate, position in eyeplates.items():
        G.add_node(eyeplate, pos=position, size=10, group='eyeplate')

    # Connect eyeplates to each other
    all_eyeplates = list(eyeplates.items())
    for i, (node1, pos1) in enumerate(all_eyeplates):
        for j, (node2, pos2) in enumerate(all_eyeplates):
            if node1 != node2:
                dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                if dist < 5:  # Connect eyeplates within a certain distance threshold
                    G.add_edge(node1, node2, weight=dist)

    # Ensure compartments are connected to only one eyeplate
    all_compartments = {**bottom_compartments, **top_compartments}
    for compartment, comp_pos in all_compartments.items():
        distances = []
        for eyeplate, eye_pos in eyeplates.items():
            dist = np.linalg.norm(np.array(comp_pos) - np.array(eye_pos))
            distances.append((dist, eyeplate))

        # Connect each compartment to its nearest eyeplate
        distances.sort()
        nearest_eyeplate = distances[0][1]
        G.add_edge(compartment, nearest_eyeplate, weight=distances[0][0])

    pos = nx.get_node_attributes(G, 'pos')
    return G, pos

# Function to visualize the 3D graph with Plotly
def visualize_3d_graph_plotly(G, pos, path=None, active_eyeplates=None):
    edge_trace = []
    path_edge_trace = []
    node_x, node_y, node_z = [], [], []
    node_text = []
    node_size = []

    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(node)
        node_size.append(G.nodes[node]['size'])
    
    # Add edges (connect nearest neighbors)
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_trace.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                                       mode='lines', line=dict(color='gray', width=2)))

    # Highlight the path in blue if given
    if path:
        path_edges = list(zip(path, path[1:]))
        for edge in path_edges:
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            path_edge_trace.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                                                mode='lines', line=dict(color='blue', width=4)))

    # Add nodes with tooltips (include active eyeplates toggle)
   

if __name__ == "__main__":
    main()

