from typing import Union
from BayesNet import BayesNet
import copy
import networkx as nx


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go

    def prune(self, variable: str) -> None:
        """
        Prunes a variable and all its descendants from the Bayesian network.
        :param variable: Variable to be pruned.
        """
        # get all descendants of the variable
        descendants = nx.descendants(self.structure, variable)
        # remove the variable and its descendants from the graph
        self.structure.remove_nodes_from(descendants + [variable])

    def d_separation(self, x: str, y: str, z: Union[str, list[str]]) -> bool:
        """
        Checks whether the variables x and y are independent given the observations in z using the d-separation criterion.
        
        :param x: The first variable to test for independence.
        :param y: The second variable to test for independence.
        :param z: The observations that are known to be true. Can be a single variable or a list of variables.
        :return: True if x and y are independent given z, False otherwise.
        """
        # create a deep copy of the BayesNet structure so we can modify it without affecting the original
        structure = copy.deepcopy(self.bn.structure)

        # if z is a single variable, convert it to a list for easier processing
        if isinstance(z, str):
            z = [z]

        # remove any edges from the copied structure that are not active given the observations z
        for var in z:
            parents = self.bn.get_parents(var)
            for parent in parents:
                structure.remove_edge(parent, var)

        # check whether x and y are independent given the modified structure
        return not nx.has_path(structure, x, y)



    def is_independent(self, X: Union[str, list[str]], Y: Union[str, list[str]], Z: Union[str, list[str]]) -> bool:
        """
        Determines whether X is independent of Y given Z.
        :param X: Set of variables X.
        :param Y: Set of variables Y.
        :param Z: Set of variables Z.
        :return: True if X is independent of Y given Z, False otherwise.
        """
        # make sure X, Y, and Z are all lists of variables
        if isinstance(X, str):
            X = [X]
        if isinstance(Y, str):
            Y = [Y]
        if isinstance(Z, str):
            Z = [Z]
            
        # make sure X, Y, and Z are all in the Bayesian network
        for var in X + Y + Z:
            if var not in self.structure.nodes():
                raise ValueError('Variable not in the Bayesian network')
                
        # make sure Z is a subset of the parents of X and Y
        for var in X + Y:
            parents = self.structure.predecessors(var)
            if not set(Z).issubset(set(parents)):
                return False

        # if all conditions are met, X is independent of Y given Z
        return True

if __name__ == '__main__':

    # Test network pruning

    # create a Bayesian network
    bn = BayesNet()

    # add some variables and edges
    bn.add_var('A')
    bn.add_var('B')
    bn.add_var('C')
    bn.add_edge('A', 'B')
    bn.add_edge('B', 'C')

    # print the network
    print(bn.structure.nodes())
    # Output: ['A', 'B', 'C']

    # prune the 'B' variable and its descendants
    bn.prune('B')

    # print the network again
    print(bn.structure.nodes())
    # Output: ['A']





    # Test d-separation
    reasoner = BNReasoner()
    # load the Bayesian network and create the BNReasoner object

    x = 'Smoke'
    y = 'Cancer'
    z = ['Pollution']

    if reasoner.d_separation(x, y, z):
        print(f'{x} and {y} are independent given {z}')
    else:
        print(f'{x} and {y} are not independent given {z}')


## test Independence
# create a Bayesian network
bn = BayesNet()

# add some variables and edges
bn.add_var('A')
bn.add_var('B')
bn.add_var('C')
bn.add_var('D')
bn.add_edge('A', 'B')
bn.add_edge('B', 'C')
bn.add_edge('B', 'D')

# check if A is independent of C given B
print(bn.is_independent('A', 'C', 'B'))
# Output: True

# check if A is independent of C given D
print(bn.is_independent('A', 'C', 'D'))
# Output: False












