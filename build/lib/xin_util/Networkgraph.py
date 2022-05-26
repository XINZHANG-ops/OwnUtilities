"""
****************************************
 * @author: Xin Zhang
 * Date: 5/19/21
****************************************
"""
import matplotlib.pyplot as plt
import networkx as nx
import random


class create_nx_graph:
    def __init__(self, relation_dict):
        """

        :param relation_dict: a dict of this form, the number is the edge weight, only 2 level dict
        {'node1' : {'node2': 1.3, 'node3': 2},
        'node2' : {'node1': 1.3, 'node4': 3, 'node5': 0.5},
        'node3': {'node2': 1, 'node4': 2},
        ...}
        """

        networkG = nx.Graph()
        for node, connect_dict in relation_dict.items():
            for neighbor_node, weight in connect_dict.items():
                networkG.add_edge(node, neighbor_node, weight=weight)
        self.networkG = networkG

    def plot(
        self,
        networkG=None,
        nodelist=None,
        node_colorlist=None,
        node_sizelist=None,
        edge_color=None,
        title=None,
        pos=None,
        Labels=True,
        show=True,
        figsize=(10, 10),
        **kwargs
    ):
        if networkG is None:
            networkG = self.networkG
        Allnodelist = list(networkG.nodes)
        if nodelist is None:
            nodelist = [Allnodelist[0]]
        types = len(nodelist)
        if node_colorlist is None:
            number_of_colors = types + 1
            node_colorlist = [
                "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)
            ]
        if node_sizelist is None:
            node_sizelist = [150] * (types + 1)

        Nodecolormap = [None] * len(Allnodelist)
        Nodesizemap = [None] * len(Allnodelist)
        for node in networkG:
            indexnode = Allnodelist.index(node)
            n = 0
            for t in range(types):
                if node in nodelist[t]:
                    Nodecolormap[indexnode] = node_colorlist[t]
                    Nodesizemap[indexnode] = node_sizelist[t]
                else:
                    n += 1
            if n == types:
                Nodecolormap[indexnode] = node_colorlist[-1]
                Nodesizemap[indexnode] = node_sizelist[-1]
        weightsDict = [networkG.get_edge_data(n1, n2) for (n1, n2) in networkG.edges]
        weightslist = [WD['weight'] for WD in weightsDict]
        fig = plt.figure(figsize=figsize)
        fig.add_subplot(1, 1, 1)
        if title is not None:
            plt.title(title)

        if pos is None:
            pos = nx.spring_layout(networkG)

        if edge_color is None:
            pass
        else:
            weightslist = [edge_color] * len(weightslist)
        nx.draw_networkx(
            networkG,
            pos=pos,
            with_labels=Labels,
            node_color=Nodecolormap,
            font_size=kwargs.get('font_size', 5),
            font_color=kwargs.get('font_color', 'black'),
            node_size=Nodesizemap,
            edge_color=kwargs.get('edge_color', weightslist),
            width=kwargs.get('width', 1),
            edge_cmap=kwargs.get('edge_cmap', plt.cm.Blues)
        )
        if show:
            plt.show()


def demo():
    nodes_dict = {
        'node1': {
            'node2': 1.3,
            'node3': 2
        },
        'node2': {
            'node1': 1.3,
            'node4': 3,
            'node5': 0.5
        },
        'node3': {
            'node2': 1,
            'node4': 2
        }
    }

    cng = create_nx_graph(nodes_dict)

    # if no nodes want to be emphasised, if node_colorlist is None, each node has a random color
    cng.plot(figsize=(8, 8), node_sizelist=[1000], font_size=20, edge_color='blue')
    cng.plot(figsize=(8, 8), node_colorlist=['blue'], font_size=20, edge_color='blue')

    # we want to emphasis node1, we give size and color for it and others
    cng.plot(
        nodelist=['node1'],
        node_colorlist=['red', 'blue'],
        node_sizelist=[1000, 100],
        figsize=(8, 8),
        font_size=20
    )

    # we want to emphasis node1 and node2, we give size and color for it and others
    cng.plot(
        nodelist=['node1', 'node2'],
        node_colorlist=['red', 'blue', 'green'],
        node_sizelist=[1000, 500, 100],
        figsize=(8, 8),
        font_size=20
    )

    # other layout
    pos = nx.spiral_layout(cng.networkG)
    cng.plot(figsize=(8, 8), node_sizelist=[1000], font_size=20, edge_color='blue', pos=pos)
