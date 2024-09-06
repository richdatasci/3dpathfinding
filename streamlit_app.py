import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import heapq
import numpy as np

# Function to generate the graph with compartments and fixed eyeplates
def generate_graph():
    G = nx.Graph()
    
    # Create four large nodes representing compartments below the eyeplates (z = -10)
    bottom_compartments = {'Compartment A': (-30, 0, -10), 
                           'Compartment B': (-20, 0, -10), 
                           'Compartment C': (-10, 0, -10), 
                           'Compartment D': (0, 0, -10)}
    
    # Create four large nodes representing compartments above the eyeplates (z = 10)
    top_compartments = {'Compartment E': (-30, 10, 10),
                        'Compartment F': (-20, 10, 10),
                        'Compartment G': (-10, 10, 10),
                        'Compartment H': (0, 10, 10)}
    
    # Add large compartment nodes (bottom and top compartments)
    for compartment, position in {**bottom_compartments, **top_compartments}.items():
        G.add_node(compartment, pos=position, size=20, group='compartment')

    # Define fixed 3D positions for 20 eyeplates (EPs)
    eyeplates = {
        'EP 1': (-25, 2.5, 1),    
        'EP 2': (-15, 2.5, -1), 
        'EP 3': (-5, 2.5, 2),    
        'EP 4': (-2, 7.5, -2),   
        'EP 5': (-28, 7.5, 3),   
        'EP 6': (-22, 3.5, 0),   
        'EP 7': (-12, 6.0, 1),
        'EP 8': (-8, 2.5, -1),
        'EP 9': (-6, 7.0, 0),
        'EP 10': (-14, 6.0, -3),
        'EP 11': (-27, 1.0, 1),
        'EP 12': (-16, 1.0, -2),
        'EP 13': (-7, 1.0, 1),
        'EP 14': (-9, 5.0, -1),
        'EP 15': (-19, 5.0, 2),
        'EP 16': (-22, 8.0, 0),
        'EP 17': (-12, 8.0, -1),
        'EP 18': (-10, 9.0, 1),
        'EP 19': (-5, 9.0, 0),
        'EP 20': (-25, 9.0, -1)
    }

    # Add eyeplate nodes with fixed positions
    for eyeplate, position in eyeplates.items():
        G.add_node(eyeplate, pos=position, size=10, group='eyeplate')

    # Connect eyeplates to each other
    all_eyeplates = list(eyeplates.items())
    for i, (node1, pos1) in enumerate(all_eyeplates):
        for j, (node2, pos2) in enumerate(all_eyeplates):
            if node1 != node2:
                dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                if dist < 10:  # Connect eyeplates within a certain distance threshold
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
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z,
                              mode='markers+text',
                              text=node_text,
                              textposition='top center',
                              marker=dict(size=node_size, color='skyblue'),
                              hoverinfo='text')

    # Camera view for landscape
    camera = dict(eye=dict(x=2.5, y=0.1, z=0.8))  # Adjust camera for a horizontal view

    fig = go.Figure(data=edge_trace + path_edge_trace + [node_trace],
                    layout=go.Layout(title='Use mouse to zoom and rotate',
                                     width=1000,  
                                     height=500,  
                                     scene_camera=camera,  
                                     showlegend=False,
                                     scene=dict(xaxis=dict(showbackground=False),
                                                yaxis=dict(showbackground=False),
                                                zaxis=dict(showbackground=False))))
    return fig

# Dijkstra's Algorithm to find the path through Eyeplates
def dijkstra_3d_with_eyeplates(graph, start, goal, active_eyeplates):
    # Filter the graph based on active eyeplates
    filtered_graph = graph.copy()
    
    # Remove inactive eyeplates
    for node in list(filtered_graph.nodes):
        if 'EP' in node and node not in active_eyeplates:
            filtered_graph.remove_node(node)

    queue = [(0, start)]
    distances = {node: float('inf') for node in filtered_graph.nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in filtered_graph.nodes}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == goal:
            break

        for neighbor, attributes in filtered_graph[current_node].items():
            weight = attributes['weight']
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    path = []
    current_node = goal
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]

    return path, distances[goal]

# Streamlit app
def main():
    st.title("3D Compartment Pathfinding")
    
    st.sidebar.header("Graph Options")
    
    # Generate the graph with compartments and eyeplates
    G, pos = generate_graph()
    
    # Select start and goal nodes (only compartments are valid start/end)
    compartments = [n for n in G.nodes if G.nodes[n]['group'] == 'compartment']
    eyeplates = [n for n in G.nodes if G.nodes[n]['group'] == 'eyeplate']
    
    start_node = st.sidebar.selectbox("Select Start Compartment:", compartments)
    goal_node = st.sidebar.selectbox("Select Goal Compartment:", compartments)

    # Allow user to turn eyeplates on/off
    active_eyeplates = st.sidebar.multiselect("Select Active Eyeplates:", eyeplates, default=eyeplates)

    if st.sidebar.button("Find Path"):
        shortest_path, shortest_distance = dijkstra_3d_with_eyeplates(G, start_node, goal_node, active_eyeplates)
        st.write(f"**Shortest path from {start_node} to {goal_node} via Eyeplates:** {shortest_path}")
        st.write(f"**Shortest distance:** {shortest_distance}")
        fig = visualize_3d_graph_plotly(G, pos, path=shortest_path, active_eyeplates=active_eyeplates)
    else:
        fig = visualize_3d_graph_plotly(G, pos, active_eyeplates=active_eyeplates)

    # Plot without container width adjustment (specific size)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
