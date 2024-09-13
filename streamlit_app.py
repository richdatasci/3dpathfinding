import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import heapq
import numpy as np
import random

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
        'EP 1': (25, 3.28, -0.07),    
        'EP 2': (4.24, 3.28, -0.07), 
        'EP 3': (-7.35, 3.28, 0.0244),    
        'EP 4': (12.12, 3.28, -0.65),   
        'EP 5': (-12.03, 3.28, -8.54),   
        'EP 6': (-11.77, 3.28, -16.639),   
        'EP 7': (-7.153, 3.28, 11.954),
        'EP 8': (0.124, 3.28, -7.119),
        'EP 9': (3.82, 3.28, -11.75),
        'EP 10': (0.006, 7.4, -12.132),
        'EP 11': (-8.775, 7.4, -12.11),
        'EP 12': (16.192, 7.4, -12.294),
        'EP 13': (-16.84, 7.4, -5.044),
        'EP 14': (-17.309, 7.4, -0.656),
        'EP 15': (-12.126, 7.4, -0.804),
        'EP 16': (-9.241, 7.4, -0.738),
        'EP 17': (-5.119, 7.4, -3.361),
        'EP 18': (-4.907, 7.4, -8.788),
        'EP 19': (2.521, 7.4, -12.71),
        'EP 20': (5.072, 7.4, -11.98),
        'EP 21': (-12.59, 11.33, -14.277),
        'EP 22': (-20.45, 11.33, -17.76),
        'EP 23': (-22.11, 11.33, 1.42),
        'EP 24': (-13.8, 11.33, 0.85),
        'EP 25': (-7.41, 11.33, 0.77),
        'EP 26': (-1.29, 11.33, 0.71),
        'EP 27': (-5.723, 11.33, -7.43)
    }

# Ensure EPs connected to compartments have a capacity of 8000kg
    compartment_connections = []
    capacities = [3000, 1600]  # Other EPs will have capacities 3000kg or 1600kg

    # Connect each compartment to its nearest eyeplate (ensure those EPs get 8000kg capacity)
    all_compartments = {**bottom_compartments, **top_compartments}
    for compartment, comp_pos in all_compartments.items():
        distances = []
        for eyeplate, eye_pos in eyeplates.items():
            dist = np.linalg.norm(np.array(comp_pos) - np.array(eye_pos))
            distances.append((dist, eyeplate))

        distances.sort()
        nearest_eyeplate = distances[0][1]
        G.add_edge(compartment, nearest_eyeplate, weight=distances[0][0])
        G.nodes[nearest_eyeplate]['capacity'] = 8000  # Assign 8000kg capacity to connected EP
        compartment_connections.append(nearest_eyeplate)

    # Assign random capacities to the remaining EPs (3000kg or 1600kg)
    for eyeplate in eyeplates:
        if eyeplate not in compartment_connections:
            G.add_node(eyeplate, pos=eyeplates[eyeplate], size=10, group='eyeplate', capacity=random.choice(capacities))
        else:
            G.add_node(eyeplate, pos=eyeplates[eyeplate], size=10, group='eyeplate')

    # Connect the remaining eyeplates to each other
    all_eyeplates = list(eyeplates.items())
    for i, (node1, pos1) in enumerate(all_eyeplates):
        for j, (node2, pos2) in enumerate(all_eyeplates):
            if node1 != node2:
                dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                if dist < 10:  # Connect eyeplates within a certain distance threshold
                    G.add_edge(node1, node2, weight=dist)

    pos = nx.get_node_attributes(G, 'pos')
    return G, pos

# Dijkstra's Algorithm with weight capacity check and vertical movement penalty
def dijkstra_3d_with_weight_and_penalty(graph, start, goal, active_eyeplates, component_weight, penalty=1.5):
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

    pos = nx.get_node_attributes(filtered_graph, 'pos')

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == goal:
            break

        for neighbor, attributes in filtered_graph[current_node].items():
            # Get the capacity of the neighbor (if it's an eyeplate)
            if 'capacity' in filtered_graph.nodes[neighbor] and filtered_graph.nodes[neighbor]['capacity'] < component_weight:
                continue  # Skip this node if the component is too heavy for the EP

            # Calculate the Euclidean distance (normal weight)
            weight = attributes['weight']

            # Apply a vertical movement penalty if there is a Z-coordinate difference
            z_diff = abs(pos[current_node][2] - pos[neighbor][2])  # Difference in Z-axis
            if z_diff > 0:
                weight += penalty * z_diff  # Apply the penalty

            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # Reconstruct the shortest path
    path = []
    current_node = goal
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]

    return path, distances[goal]

# Function to visualise the 3D graph
def visualize_3d_graph_plotly(G, pos, path=None, active_eyeplates=None):
    edge_trace = []
    path_edge_trace = []
    node_x, node_y, node_z = [], [], []
    node_text = []
    node_size = []
    node_capacity = []

    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(f"{node} (R: {G.nodes[node]['capacity']}kg)" if 'capacity' in G.nodes[node] else node)
        node_size.append(G.nodes[node]['size'])
        if 'capacity' in G.nodes[node]:
            node_capacity.append(G.nodes[node]['capacity'])

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
                                     width=1500,  # Set the width to 2500 pixels
                                     height=1000,  # Set the height to 1500 pixels
                                     scene_camera=camera,  # Apply the camera for landscape view
                                     showlegend=False,
                                     scene=dict(xaxis=dict(showbackground=False),
                                                yaxis=dict(showbackground=False),
                                                zaxis=dict(showbackground=False))))
    return fig

# Streamlit app
def main():
    st.title("3D Compartment Pathfinding")

    # Component selection
    st.sidebar.header("Select Component")
    components = {
        'Distillation Pump (2000kg)': 2000,
        'Battery (500kg)': 500,
        'Air Pump (1000kg)': 1000,
        'Thermal Shield (100kg)': 100
    }
    selected_component = st.sidebar.selectbox("Choose a component:", list(components.keys()))
    component_weight = components[selected_component]

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
        shortest_path, shortest_distance = dijkstra_3d_with_weight_and_penalty(G, start_node, goal_node, active_eyeplates, component_weight)
        st.write(f"**Shortest path from {start_node} to {goal_node} via Eyeplates:** {shortest_path}")
        st.write(f"**Shortest distance:** {shortest_distance}")
        fig = visualize_3d_graph_plotly(G, pos, path=shortest_path, active_eyeplates=active_eyeplates)
    else:
        fig = visualize_3d_graph_plotly(G, pos, active_eyeplates=active_eyeplates)

    # Plot without container width adjustment (specific size)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()

