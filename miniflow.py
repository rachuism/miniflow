import numpy as np

def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def forward_pass(output_node, sorted_nodes):

    for n in sorted_nodes:
        n.forward() #necesario ejecutarlo nodo por nodo
    
    return output_node.value

def forward_pass_graph(graph):
    """
    Performs a forward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()


class Node(object):
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes

        self.outbound_nodes = []

        for n in self.inbound_nodes: #Estoy iterando sobre el resto de objetos
            n.outbound_nodes.append(self)

        self.value = None #Hasnt been set yet

    def forward(self):

        raise NotImplemented

class Input(Node):
    def __init__(self):

        Node.__init__(self)

    def forward(self, value=None): #Tengo que introducir el valor yo

        if value is not None:
            self.value = value

class Add(Node):
    def __init__(self, *inputs): #Repacks the argumnents into a tuple
        Node.__init__(self, inputs) #[x, y] son los nodos que estoy sumando

    def forward(self):
        sum = 0
        for i in self.inbound_nodes:
            
            sum = sum + i.value
        self.value = sum

class Mul(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        mul = self.inbound_nodes[0].value
        for i in self.inbound_nodes[1:]:

            mul = mul*i.value
        self.value = mul

class Linear(Node):
    def __init__(self, X, W, b):
        # Notice the ordering of the input nodes passed to the
        # Node constructor.
        Node.__init__(self, [X, W, b])

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b


class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):

        return 1. / (1. + np.exp(-x))

    def forward(self):

        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)

class MSE(Node):
    def __init__(self, y, a):

        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        shape = len(y)
        x = 0
        squares = 0
        while x < (shape):
            squares += np.square(y[x]-a[x])
            x += 1

        self.value = squares / shape



    


            

