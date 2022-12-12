from typing import Union
from BayesNet import BayesNet
import copy
import networkx as nx
import pandas as pd


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

    def is_path_activedd(self, start, middle, end, evidence):
        #Causal
        if middle in bn.get_children(start) and end in bn.get_children(middle):
            if middle in evidence:
                return False
        
        #Fork
        if start in bn.get_children(middle) and end in bn.get_children(end):
            if middle in evidence:
                return False

        #Collider
        if middle in bn.get_children(start) and middle in bn.get_children(end):
            if middle not in evidence:
                return False
        
        return True

    def multiply_cpts(self, cpt1, cpt2)

    def prune(self, query: list, evidence: dict) -> None:
        """
        Prunes a variable and all its descendants from the Bayesian network.
        :param variable: Variable to be pruned.
        """
        mod = True
        while mod:
            mod = False
            for var in self.bn.get_all_variables():
                if self.bn.get_children(var) == [] and var not in evidence and var not in query:
                    self.bn.del_var(var)
                    mod = True

            cpts = self.bn.get_all_cpts()
            for var, value in evidence.items():
                for node in self.bn.get_all_variables():
                    cpt = cpts[node]
                    if var in cpt.columns:
                        idxs = cpt[cpt[var] != value].index
                        cpt = cpt.drop(idxs)
                        self.bn.update_cpt(var,cpt)

                descendants = self.bn.get_children(var)   
                for node in descendants:
                    self.bn.del_edge((var, node))
                    mod = True


    def d_separation(self, x: str, y: str, z: list) -> bool:
        """
        Checks whether the variables x and y are independent given the observations in z using the d-separation criterion.
        
        :param x: The first variable to test for independence.
        :param y: The second variable to test for independence.
        :param z: The observations that are known to be true. Can be a single variable or a list of variables.
        :return: True if x and y are independent given z, False otherwise.
        """

        paths = nx.all_simple_paths(self.bn.structure, x, y)
        for path in paths:
            active = True
            print(path)
            for idx in range(1, len(path)-1):
                if not self.is_path_activedd(path[idx-1], path[idx], path[idx+1], z):
                    active = False
                    break
            if active == True:
                break

        
        return not active

    def is_independent(self, x: str, y: str, z: list) -> bool:
        """
        Determines whether X is independent of Y given Z.
        :param X: Set of variables X.
        :param Y: Set of variables Y.
        :param Z: Set of variables Z.
        :return: True if X is independent of Y given Z, False otherwise.
        """
        if self.d_separation(x,y,z):
            return True
        else: 
            return False


    

if __name__ == '__main__':
    bn = BayesNet()
    bn.load_from_bifxml('Projects/KR/Project-BN/testing/lecture_example.BIFXML')
    bnr = BNReasoner(bn)
    bn.draw_structure()
    bnr.prune([],{'Rain?':False})
    bn.draw_structure()









