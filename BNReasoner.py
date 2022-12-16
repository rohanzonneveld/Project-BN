from typing import Union
from BayesNet import BayesNet
import copy
import networkx as nx
import pandas as pd
from itertools import combinations
import sys


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

    def count_nodes_and_edges(self):
        edges = 0
        nodes = 0
        for edge in self.bn.structure.edges:
            edges += 1
        for node in self.bn.structure.nodes:
            nodes += 1
        
        return nodes, edges, nodes + edges
    def is_path_active(self, start, middle, end, evidence):
        #Causal
        if (middle in self.bn.get_children(start) and end in self.bn.get_children(middle)) or (middle in self.bn.get_children(end) and start in self.bn.get_children(middle)):
            if middle in evidence:
                return False
        
        #Fork
        if start in self.bn.get_children(middle) and end in self.bn.get_children(middle):
            if middle in evidence:
                return False

        #Collider
        if middle in self.bn.get_children(start) and middle in self.bn.get_children(end):
            if not self.test_node_in_evidence(middle, evidence):
                return False

        
        return True

    def test_node_in_evidence(self, node, evidence) -> bool:
        if node not in evidence:
            children = self.bn.get_children(node)
            for child in children:
                if self.test_node_in_evidence(child, evidence):
                    return True
            return False
        return True   



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
        cp = copy.deepcopy(self)
        for edge in self.bn.structure.edges:
            cp.bn.structure.add_edge(edge[1], edge[0])

        paths = nx.all_simple_paths(cp.bn.structure, x, y)
        for path in paths:
            active = True
            for idx in range(1, len(path)-1):
                if not self.is_path_active(path[idx-1], path[idx], path[idx+1], z):
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

    def maxing_out(self, CPT: dict, variable: str):
        """
        function to maximize out a variable from a cpt
        input:  - a cpt (dict)
                - the variable to maximize out from the cpt (str)
        output: the maximized out cpt containing the info of the maximized out variable
        """
        new_CPT = pd.DataFrame({})
        vars = [x for x in CPT if x not in ['p', 'maxed', variable]]

        # loop over all possibilities for X\variable
        # loop over every row in cpt
        for i in range(len(CPT)):
            # select instantiation from line i
            instantiation = pd.Series(dict(zip(vars,CPT.iloc[i][vars])))
            # get all compatible instantiations
            table = self.bn.get_compatible_instantiations_table(instantiation, CPT)
            # select row with highest p-value
            p_value = table['p'].max()
            # get index of maximum instantiation
            # idx = table[table==p_value].index[0]
            idx = list(table['p']).index(p_value)
            # add assignment of variable to row
            line = table.iloc[idx].copy()
            truth = table.iloc[idx][variable]
            if 'maxed' in line.keys():
                line['maxed'] += f', {variable}={truth}'
            else:
                line['maxed'] = f'{variable}={truth}'
            # add new maxed out line to new cpt
            new_CPT = new_CPT.append(line, 
                        ignore_index=True)
            # delete maximized out variable from cpt
            new_CPT.pop(variable)
            
        return new_CPT.drop_duplicates()

    def factor_mul(self, factor1, factor2):
        """
        function to perform factor multiplication between n factors
        Input: dict of CPTS
        Output: The CPT resulting from the factor multiplication
        """
        
        results = pd.DataFrame({})
                
        # get shared vars from factor 1
        shared_vars = [x for x in factor1.keys() if x in factor2.keys() and x not in ['p', 'maxed']]
        if len(shared_vars) == 0:
            print('Cannot mulitply factors because they do not contain overlapping variables')
            sys.exit()

        # get vars from factor 2 which are to be added to factor 1
        added_vars = [x for x in factor2.keys() if x not in factor1.keys()]

        # loop over rows from factor 1
        for row1 in range(len(factor1)):
            # for every row  in factor 1 get the instantiation of the shared vars
            instantiation = pd.Series(dict(zip(shared_vars,factor1.iloc[row1][shared_vars])))
            # get the lines from factor 1 compatible with current instantiation
            compats1 = self.bn.get_compatible_instantiations_table(instantiation, factor1)
            # loop to create multiplication of factors over compatible rows
            for row2 in range(len(factor2)):
                # for every row in factor 2 get the instantiation of the shared vars
                instantiation = pd.Series(dict(zip(shared_vars,factor2.iloc[row2][shared_vars])))
                # get the lines from factor 2 compatible with the current instantiation
                compats2 = self.bn.get_compatible_instantiations_table(instantiation, factor2)
                # loop over compatible rows from both factor1 and factor2
                for i in range(len(compats1)):
                    line1 = compats1.iloc[i]
                    for j in range(len(compats2)):
                        line2 = compats2.iloc[j]
                        # multiply p values
                        p = line1['p']*line2['p']
                        # copy the line from factor1
                        result = line1.copy()
                        # add all vars not in factor1 to result
                        for var in added_vars:
                            result[var] = line2[var]
                        # update p value from new row
                        result['p'] = p
                        # append new row to output
                        results=results.append(result, 
                                ignore_index=True)
        
            CPT = results.drop_duplicates()
            return CPT
        
    def cpt_mul(self, cpt1, cpt2):
        new_CPT = pd.DataFrame()
        columns2 = list(cpt2)
        columns1 = list(cpt1)
        
        for i in range(len(cpt1)):
            for j in range(len(cpt2)):
                p_1 = cpt1.iloc[i]["p"]
                p_2 = cpt2.iloc[j]["p"]

                new_p = round(p_1 * p_2, 8)

                clean = cpt1.iloc[i].drop(['p'])

                new_row = clean.append(cpt2.iloc[j])

                new_row["p"] = new_p

                new_CPT = new_CPT.append(new_row, ignore_index=True)
        final_cpt = new_CPT[new_CPT['p'] != 0]

        return final_cpt

    def mindeg_order(self, X: list):
        """
        function to decide in what order to eliminate the variables for a query based on the 
        least amount of existing edges between nodes in the interactiongraph
        input:  - a list X of variables to be eliminated
        output: - a sorted list of variables X in the order to be eliminated
        """
        interaction_graph = self.bn.get_interaction_graph()
        nodes = {}
        pi = []
        
        # count the amount of neighbors for all variables
        for var in X:
            nodes[var] = len(list(interaction_graph.neighbors(var)))
        
        for _ in range(len(X)):
            # get the var with the least edges
            next_var = min(nodes.keys(), key= lambda var: nodes[var])
            nodes.pop(next_var)
            # append this var in the elimination order
            pi.append(next_var)

            # get a list of all neighbors of var
            neighbors = list(interaction_graph.neighbors(next_var))
            # remove node from interaction graph
            interaction_graph.remove_node(next_var)
            # loop over all possible pairs of neighbor and add a node if there isn't already
            for pair in list(combinations(neighbors, 2)):
                if not interaction_graph.has_edge(pair[0], pair[1]):
                    interaction_graph.add_edge(pair[0], pair[1])

        return pi

    def minfil_order(self, X: list):
        """
        function to decide in what order to eliminate the variables for a query based on 
        adding the least amount of new edges between nodes in the interactiongraph as a 
        result of summing out the variable

        input:  - a list X of variables to be eliminated
        output: - a sorted list of variables X in the order to be eliminated
        """

        interaction_graph = self.bn.get_interaction_graph()
        pi = []
        new_edges = {var: 0 for var in X}

        for _ in range(len(X)):
            # count how many edges the removal of each var in X would add
            for var in new_edges.keys():
                neighbors = list(interaction_graph.neighbors(var))
                for pair in list(combinations(neighbors, 2)):
                    if not interaction_graph.has_edge(pair[0], pair[1]):
                        new_edges[var] += 1 
            next_var = min(new_edges.keys(), key = lambda var: new_edges[var])
            pi.append(next_var)
            new_edges.pop(next_var)

        return pi
    
    def summing_out(self, CPT, index_same, list):
        '''
        create the final new CPT without the variables that should be summed out
        '''
        new_CPT = CPT.copy()
        for variable in list:
            new_CPT = new_CPT.drop(columns=[variable])

        for key in index_same:

            p_value_sum = CPT.iloc[key]['p']

            equal_indexes = [index_same.get(key)]

            for i in equal_indexes:
                p = CPT.iloc[i]['p'].values
                p_value_sum += p[0]
                new_CPT.at[i, 'p'] = 0

            new_CPT.at[key, 'p'] = p_value_sum

        for index_CPT in range(len(new_CPT)):
            if new_CPT.iloc[index_CPT]['p'] == 0:
                new_CPT.drop([index_CPT])

        return new_CPT
    
    def Variable_elimination(self, CPT, list, type):

        index_same = {}

        clean_CPT = CPT.copy()

        clean_CPT = clean_CPT.drop(columns=["p"])
        for variable in list:
            clean_CPT = clean_CPT.drop(columns=[variable])

        for row_1 in clean_CPT.iloc:
            for i in range(row_1.name + 1, len(clean_CPT)):
                row_2 = clean_CPT.iloc[i]

                if row_1.equals(row_2):

                    if row_1.name in index_same:
                        index_same[row_1.name].append(i)
                    else:
                        index_same[row_1.name] = [i]

        if type == "sum":
            new_CPT = self.summing_out(CPT, index_same, list)
        elif type == "max":
            new_CPT = self.maxing_out(CPT, index_same)

        return new_CPT
    
    def marginal_distribution(self, Q, evidence, order):
         '''
         Q = variables in the network BN
         evidence = instantiation of some variables in the BN

         output = posterior marginal Pr(Q|e)
         '''

         self.prune(Q, evidence)
         not_Q = [x for x in self.bn.get_all_variables() if x not in Q]
         if order == 1:
             order = set(self.mindeg_order(not_Q))
         elif order == 2:
             order = set(self.minfil_order(not_Q))

         # order = set(self.randomOrder(self.bn.get_all_variables()))
         # order_no_Q = order.difference(Q)

         # get all cpts eliminating rows incompatible with evidence
         for ev in evidence.keys():

             cpts = self.bn.get_cpt(ev)
             cpts = self.bn.reduce_factor(evidence, cpts)

             for row in range(len(cpts)):
                 if cpts.iloc[row]['p'] == 0:
                     cpts.drop([row])

             self.bn.update_cpt(ev, cpts)

             for child in self.bn.get_children(ev):
                 cpts = self.bn.get_cpt(child)
                 cpts = self.bn.reduce_factor(evidence, cpts)

                 for row in range(len(cpts)):
                     if cpts.iloc[row]['p'] == 0:
                         cpts.drop([row])

                 self.bn.update_cpt(child, cpts)

         # make CPTs of all variables in Pi not in Q
         S = list(self.bn.get_all_cpts().values())

         for variable in order:
             list_cpts = []
             list_goed = []

             for i in range(0, len(S)):
                 columns = list(S[i])
                 if variable in columns:
                     list_cpts.append(S[i])
                 else:
                     list_goed.append(S[i])

             if len(list_cpts) > 0:
                 cpt1 = list_cpts[0]

             if len(list_cpts) == 1:
                 list_goed.append(list_cpts[0])

             if len(list_cpts) > 1:
                 for cpt2 in list_cpts[1:]:
                     cpt1 = self.factor_mul(cpt1, cpt2)
                 final_cpt = cpt1

                 factor = self.summing_out(final_cpt, [variable], 'sum')
                 list_goed.append(factor)

             S = list_goed
         cpt_new=pd.DataFrame()
         for i in range(0, len(S) - 1):
             if len(set(list(S[i])).intersection(set(list(S[i])))) > 1:
                 cpt_new = self.factor_mul(S[i], S[i + 1])
             else:
                 cpt_new = self.factor_mul(S[i], S[i + 1])
             S[i + 1] = cpt_new
         final_cpt = cpt_new
         final_cpt = final_cpt[final_cpt['p'] != 0]

         for var in list(cpt_new):
             if var != "p":
                 if var not in Q:
                     final_cpt = self.reduce_factor(final_cpt, [var], 'sum')

         normalize_factor = final_cpt['p'].sum()
         final_cpt['p'] = final_cpt['p'] / normalize_factor

         # zero
         final_cpt = final_cpt[final_cpt['p'] != 0]

         return final_cpt

    def MAP_MPE(self, Q, evidence, order):

         # if it is a MPE than map_var is the total BN with all the possible values in it
         pruned = self.prune(Q, evidence)

         variables = self.bn.get_all_variables()

         if list(set(variables) - set(Q)) == []:
             not_Q = variables
         else:
             not_Q = list(set(variables) - set(Q))

         if order == 1:
             order = set(self.mindeg_order(not_Q))
         elif order == 2:
             order = set(self.minfil_order(not_Q))


         # add MAP variables last
         if len(Q) != 0:
             for i in Q:
                 order.add(i)

         # get all cpts eliminating rows incompatible with evidence
         for ev in evidence.keys():

             cpts = self.bn.get_cpt(ev)
             cpts = self.bn.reduce_factor(evidence, cpts)

             for row in range(len(cpts)):
                 if cpts.iloc[row]['p'] == 0:
                     cpts.drop([row])

             self.bn.update_cpt(ev, cpts)

             for child in self.bn.get_children(ev):
                 cpts = self.bn.get_cpt(child)
                 cpts = self.bn.reduce_factor(evidence, cpts)

                 for row in range(len(cpts)):
                     if cpts.iloc[row]['p'] == 0:
                         cpts.drop([row])

                 self.bn.update_cpt(child, cpts)

         # make CPTs of all variables in Pi not in Q
         S = list(self.bn.get_all_cpts().values())

         for variable in order:

             list_cpts = []
             list_goed = []

             for i in range(0, len(S)):
                 columns = list(S[i])
                 if variable in columns:
                     list_cpts.append(S[i])
                 else:
                     list_goed.append(S[i])

             if len(list_cpts) == 1:
                 list_goed.append(list_cpts[0])

             if len(list_cpts) > 0:
                 cpt1 = list_cpts[0]

             if len(list_cpts) > 1:
                 for cpt2 in list_cpts[1:]:
                     cpt1 = self.factor_mul(cpt1, cpt2)

                 final_cpt = cpt1
                 final_cpt = final_cpt[final_cpt['p'] != 0]

                 if Q == {} or variable in Q:

                     cpt = self.maxing_out(final_cpt, variable)

                 else:
                     cpt = self.reduce_factor(final_cpt, [variable], 'sum')

                 cpt = cpt[cpt['p'] != 0]

                 list_goed.append(cpt)

             S = list_goed

             # print(S)
         if len(S) == 1:
             cpt_new = S[0]
         else:
             for i in range(0, len(S) - 1):
                 cpt_new = self.factor_mul(S[i], S[i + 1])

                 S[i + 1] = cpt_new
         final_cpt = cpt_new

         highest_p = 0
         final_cpt = final_cpt[final_cpt['p'] != 0]

         length_final_cpt = len(final_cpt)
         for i in range(length_final_cpt):

             if final_cpt.iloc[i]["p"] > highest_p:
                 highest_p = final_cpt.iloc[i]["p"]
                 final_row = final_cpt.iloc[i]

             # else:
             #     final_cpt.iloc[i]["p"] = 0
             # final_cpt = final_cpt.drop(axis=1,index=[i])

         newest_cpt = pd.DataFrame()
         newest_cpt = newest_cpt.append(final_row, ignore_index=True)

         for column in newest_cpt:
             if column != "p":
                 if column not in Q and Q != {}:
                     newest_cpt = newest_cpt.drop(columns=[column])

         # print(f"final_cpt {newest_cpt}")
         return newest_cpt
        
     

