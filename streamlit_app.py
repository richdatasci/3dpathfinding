import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import heapq
import numpy as np
import json

# Load the Eyeplate data from the provided JSON file
def load_eyeplate_data():
    with open('EyeplateComms.json') as f:
        data = json.load(f)
    return data

# Function to generate the graph with eyeplates connected by floors
def generate_graph_by_floors(eyeplates_data):
    G = nx.Graph()

    # Sort eyeplates by Y coordinate to categorize floors
    floors = {}
    for eyeplate in eyeplates_data:
        y_coord = eyeplate['Y']
        eyeplate_name = eyeplate['eyeplateId']
        position = (eyeplate['X'], eyeplate['Y'], eyeplate['Z'])
        if y_coord not in floors:
            floors[y_coord] = []
        floors[y_coord].append((eyeplate_name, position))

    # Connect all eyeplates within the same floor
    for y_coord, eyeplates in floors.items():
        for i, (eyeplate_name_1, pos1) in enumerate(eyeplates):
            G.add_node(eyeplate_name_1, pos=pos1, size=10, group='eyeplate', capacity=np.random.choice([3000, 1600]))
            for j, (eyeplate_name_2, pos2) in enumerate(eyeplates):
                if i != j:
                    dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    G.add_edge(eyeplate_name_1, eyeplate_name_2, weight=dist)

    # Connect one eyeplate from each floor to the floor above
    sorted_floors = sorted(floors.keys())
    for i in range(len(sorted_floors) - 1):
        floor_current = sorted_floors[i]
        floor_above = sorted_floors[i + 1]
        # Connect the first eyeplate of the current floor to the first eyeplate of the floor above
        eyeplate_current = floors[floor_current][0][0]
        eyeplate_above = floors[floor_above][0][0]
        pos_current = floors[floor_current][0][1]
        pos_above = floors[floor_above][0][1]
        dist = np.linalg.norm(np.array(pos_current) - np.array(pos_above))
        G.add_edge(eyeplate_current, eyeplate_above, weight=dist)

    pos = nx.get_node_attributes(G, 'pos')
    return G, pos
    
# Dijkstra's Algorithm to find the most efficient and second most efficient paths
def dijkstra_3d_top_2_paths(graph, start, goal, active_eyeplates, component_weight, penalty=1.5):
    # Filter the graph based on active eyeplates
    filtered_graph = graph.copy()

    # Remove inactive eyeplates
    for node in list(filtered_graph.nodes):
        if 'EP' in node and node not in active_eyeplates:
            filtered_graph.remove_node(node)

    pos = nx.get_node_attributes(filtered_graph, 'pos')

    # Run Dijkstra's algorithm for the most efficient path
    queue = [(0, start)]
    distances = {node: float('inf') for node in filtered_graph.nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in filtered_graph.nodes}

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

    # Reconstruct the most efficient path
    first_path = []
    current_node = goal
    while current_node is not None:
        first_path.insert(0, current_node)
        current_node = previous_nodes[current_node]

    # Instead of removing edges, increase their weight to make them less favorable for the second path
    first_path_edges = list(zip(first_path, first_path[1:]))

    for edge in first_path_edges:
        if filtered_graph.has_edge(*edge):
            filtered_graph[edge[0]][edge[1]]['weight'] *= 10  # Increase the weight of the edge

    # Run Dijkstra's algorithm for the second path
    queue = [(0, start)]
    distances = {node: float('inf') for node in filtered_graph.nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in filtered_graph.nodes}

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

    # Reconstruct the second most efficient path
    second_path = []
    current_node = goal
    while current_node is not None:
        second_path.insert(0, current_node)
        current_node = previous_nodes[current_node]

    return first_path, second_path

# Function to save both routes to a JSON file
def save_routes_to_json(start, goal, component, first_path, second_path):
    route_data = {
        "start": start,
        "goal": goal,
        "component": {
            "name": component[0],
            "weight": component[1]
        },
        "most_efficient_path": first_path,
        "second_most_efficient_path": second_path
    }
    # Save to route_output.json
    with open('route_output.json', 'w') as f:
        json.dump(route_data, f, indent=4)
    st.write("**Routes saved to `route_output.json`**")

# Function to visualize the 3D graph with Plotly (highlight most efficient path in green, second in blue)
def visualize_3d_graph_plotly(G, pos, first_path=None, second_path=None, active_eyeplates=None):
    edge_trace = []
    first_path_edge_trace = []
    second_path_edge_trace = []
    node_x, node_y, node_z = [], [], []
    node_text = []
    node_size = []
    node_capacity = []

    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(f"{node} (Capacity: {G.nodes[node]['capacity']}kg)" if 'capacity' in G.nodes[node] else node)
        node_size.append(G.nodes[node]['size'])
        if 'capacity' in G.nodes[node]:
            node_capacity.append(G.nodes[node]['capacity'])

    # Add edges (connect nearest neighbors)
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_trace.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                                       mode='lines', line=dict(color='gray', width=2)))

    # Highlight the most efficient path in green
    if first_path:
        first_path_edges = list(zip(first_path, first_path[1:]))
        for edge in first_path_edges:
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            first_path_edge_trace.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                                                      mode='lines', line=dict(color='green', width=6)))

    # Highlight the second most efficient path in blue
    if second_path:
        second_path_edges = list(zip(second_path, second_path[1:]))
        for edge in second_path_edges:
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            second_path_edge_trace.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                                                       mode='lines', line=dict(color='blue', width=4)))

    # Add nodes with tooltips (include active eyeplates toggle)
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z,
                              mode='markers+text',
                              text=node_text,
                              textposition='top center',
                              marker=dict(size=node_size, color='skyblue'),
                              hoverinfo='text')

    # Adjust camera for portrait view (rotate to see levels better)
    camera = dict(eye=dict(x=0.1, y=2.5, z=2.0))  # Adjust the x, y, z for a better portrait perspective

    # Adjust the width and height to fit portrait (taller than wider)
    fig = go.Figure(data=edge_trace + first_path_edge_trace + second_path_edge_trace + [node_trace],
                    layout=go.Layout(title='Use mouse to zoom and rotate (Portrait Orientation)',
                                     width=1000,  # Portrait width
                                     height=1500,  # Portrait height (taller)
                                     scene_camera=camera,  # Apply the camera for portrait view
                                     showlegend=False,
                                     scene=dict(xaxis=dict(showbackground=False),
                                                yaxis=dict(showbackground=False),
                                                zaxis=dict(showbackground=False))))
    return fig

# Streamlit app
def main():
    st.title("3D Pathfinder")

    # Load eyeplate data from JSON
    eyeplates_data = load_eyeplate_data()

    # Component selection
    st.sidebar.header("Select Component")
    components = {
        'Distillation Pump (2000kg)': 2000,
        'Battery (500kg)': 500,
        'Air Pump (1000kg)': 1000,
        'Thermal Shield (100kg)': 100
    }
    selected_component_name = st.sidebar.selectbox("Choose a component:", list(components.keys()))
    component_weight = components[selected_component_name]

    st.sidebar.header("Graph Options")
    
    # Generate the graph with eyeplates connected by floors
    G, pos = generate_graph_by_floors(eyeplates_data)
    
    # Select start and goal nodes (eyeplates are valid start/end)
    eyeplates = [n for n in G.nodes if G.nodes[n]['group'] == 'eyeplate']
    
    start_node = st.sidebar.selectbox("Select Start Eyeplate:", eyeplates)
    goal_node = st.sidebar.selectbox("Select Goal Eyeplate:", eyeplates)

    # Allow user to turn eyeplates on/off
    active_eyeplates = st.sidebar.multiselect("Select Active Eyeplates:", eyeplates, default=eyeplates)

    if st.sidebar.button("Find Path"):
        # Run the pathfinding algorithm to get the most efficient and second most efficient paths
        first_path, second_path = dijkstra_3d_top_2_paths(G, start_node, goal_node, active_eyeplates, component_weight)

        # Output the results
        st.write(f"**Most efficient path from {start_node} to {goal_node}:** {first_path}")
        st.write(f"**Second most efficient path from {start_node} to {goal_node}:** {second_path}")

        # Save the routes to a JSON file
        save_routes_to_json(start_node, goal_node, (selected_component_name, component_weight), first_path, second_path)

        # Visualize the most efficient route in green and the second most efficient route in blue
        fig = visualize_3d_graph_plotly(G, pos, first_path=first_path, second_path=second_path, active_eyeplates=active_eyeplates)
    else:
        fig = visualize_3d_graph_plotly(G, pos, active_eyeplates=active_eyeplates)

    # Plot the 3D visualization in the app
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()

           
