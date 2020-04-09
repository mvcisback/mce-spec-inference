import networkx as nx


def label_edge(graph, edge):
    data = graph.edges()[edge]

    # Note: Sometimes entries purposely set to None.
    label = ""
    if data.get('prob') is not None:
        label += f'p={data["prob"]:.2f}\n'
    if data.get('action') is not None:
        label += f'a={data["action"]}\n'
    if data.get('visitation') is not None:
        label += f'f={data["visitation"]}\n'

    data['label'] = label


def label_node(graph, node):
    data = graph.nodes()[node]
    if data['decision'] or isinstance(data['var'], bool):
        data['label'] = f"var={data['var']}\nlvl={data['lvl']}"


def draw(graph, path):
    for edge in graph.edges():
        label_edge(graph, edge)

    for node in graph.nodes():
        label_node(graph, node)

    # convert to a graphviz agraph
    nx.drawing.nx_pydot.write_dot(graph, path)
