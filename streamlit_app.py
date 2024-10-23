import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import heapq
import numpy as np
import json

# Load the Eyeplate data from the provided JSON file
def load_eyeplate_data():
    with open('EyeplateComms.json') as f: # This can be changed to what ever local path we need
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

# Dijkstra's Algorithm to find top 2 paths
def dijkstra_3d_top_2_paths(graph, start, goal, active_eyeplates, component_weight, penalty=1.5):
    # Filter the graph based on active eyeplates
    filtered_graph = graph.copy()

    # Remove inactive eyeplates
    for node in list(filtered_graph.nodes):
        if 'EP' in node and node not in active_eyeplates:
            filtered_graph.remove_node(node)

    pos = nx.get_node_attributes(filtered_graph, 'pos')
    
    # Run Dijkstra's algorithm for the first (most efficient) path
    queue = [(0, start)]
    distances = {node: float('inf') for node in filtered_graph.nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in filtered_graph.nodes}

    first_path_edges = []
    
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

    # Reconstruct the most efficient path (first path)
    first_path = []
    current_node = goal
    while current_node is not None:
        first_path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    
    # Remove edges from the first path and recalculate for the second path
    first_path_edges = list(zip(first_path, first_path[1:]))

    for edge in first_path_edges:
        if filtered_graph.has_edge(*edge):
            filtered_graph.remove_edge(*edge)

    # Run Dijkstra's algorithm for the second path
    queue = [(0, start)]
    distances = {node: float('inf') for node in filtered_graph.nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in filtered_graph.nodes}

    second_path_edges = []
    
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

    # Reconstruct the second most efficient path (second path)
    second_path = []
    current_node = goal
    while current_node is not None:
        second_path.insert(0, current_node)
        current_node = previous_nodes[current_node]

    return first_path, second_path

# Streamlit app
def main():
    st.title("3D Ship Compartment Pathfinding Visualization")

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
    selected_component = st.sidebar.selectbox("Choose a component:", list(components.keys()))
    component_weight = components[selected_component]

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
        # Run the pathfinding algorithm to get the top 2 paths
        first_path, second_path = dijkstra_3d_top_2_paths(G, start_node, goal_node, active_eyeplates, component_weight)

        # Output the results
        st.write(f"**Most efficient path from {start_node} to {goal_node}:** {first_path}")
        st.write(f"**Second most efficient path from {start_node} to {goal_node}:** {second_path}")

        # Visualise the top 2 routes in the 3D graph
        fig = visualize_3d_graph_plotly(G, pos, first_path=first_path, second_path=second_path, active_eyeplates=active_eyeplates)
    else:
        fig = visualize_3d_graph_plotly(G, pos, active_eyeplates=active_eyeplates)

    # Plot the 3D visualization in the app
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
