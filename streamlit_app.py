import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import string
import random
import heapq


# generate the graph
def generate_graph():
    alphabet = list(string.ascii_uppercase)
    node_labels = alphabet + ['A' + letter for letter in alphabet[:22]]

    G = nx.Graph()
    G.add_nodes_from(node_labels)

    for _ in range(94):
        node1, node2 = random.sample(node_labels, 2)
        weight = random.randint(1, 10)
        G.add_edge(node1, node2, weight=weight)

    pos = {node: (random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)) for node in node_labels}

    return G, pos


# visualise the 3D graph
def visualize_3d_graph_plotly(G, pos, path=None):
    edge_trace = []
    path_edge_trace = []
    node_x, node_y, node_z = [], [], []
    node_text = []

    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(node)

    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_trace.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                                       mode='lines', line=dict(color='gray', width=2)))

    if path:
        path_edges = list(zip(path, path[1:]))
        for edge in path_edges:
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            path_edge_trace.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                                                mode='lines', line=dict(color='blue', width=4)))

    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z,
                              mode='markers+text',
                              text=node_text,
                              textposition='top center',
                              marker=dict(size=8, color='skyblue'),
                              hoverinfo='text')

    fig = go.Figure(data=edge_trace + path_edge_trace + [node_trace],
                    layout=go.Layout(title='Use mouse to rotate and zoom visual',
                                     showlegend=False,
                                     width=1000,
                                     height=800,
                                     scene=dict(xaxis=dict(showbackground=False),
                                                yaxis=dict(showbackground=False),
                                                zaxis=dict(showbackground=False))))
    return fig


# Dijkstra's Algorithm implementation
def dijkstra_3d(graph, start, goal):
    queue = [(0, start)]
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in graph.nodes}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_node == goal:
            break

        for neighbor, attributes in graph[current_node].items():
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
    st.title("3D Graph Path Finding Demo")

    st.sidebar.header("Graph Options")
    G, pos = generate_graph()

    nodes = list(G.nodes)
    start_node = st.sidebar.selectbox("Select Start Point:", nodes)
    goal_node = st.sidebar.selectbox("Select Goal Point:", nodes)

    if st.sidebar.button("Run Algorithm"):
        shortest_path, shortest_distance = dijkstra_3d(G, start_node, goal_node)
        st.write(f"**Shortest path from {start_node} to {goal_node}:** {shortest_path}")
        st.write(f"**Shortest distance:** {shortest_distance}")
        fig = visualize_3d_graph_plotly(G, pos, path=shortest_path)
    else:
        fig = visualize_3d_graph_plotly(G, pos)

    st.plotly_chart(fig)


if __name__ == "__main__":
    main()