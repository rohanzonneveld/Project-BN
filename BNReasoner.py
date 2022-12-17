from typing import Union, List, Set
from BayesNet import BayesNet
import copy
import networkx as nx
import pandas as pd
import itertools
import sys
import random



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
            for node,value in evidence.items():
                descendants = self.bn.get_children(node)
                for child in descendants:
                    self.bn.del_edge((node,child))
                    mod = True
            
            for node in self.bn.get_all_variables():
                if node not in evidence and node not in query:
                    if not self.bn.get_children(node):
                        self.bn.del_var(node)
                        mod = True

    # @profile                 
    def d_separation(self, x: str, y: str, z: list) -> bool:
        """
        Checks whether the variables x and y are independent given the observations in z using the d-separation criterion.
        
        :param x: The first variable to test for independence.
        :param y: The second variable to test for independence.
        :param z: The observations that are known to be true. Can be a single variable or a list of variables.
        :return: True if x and y are independent given z, False otherwise.
        """
        # cp = copy.deepcopy(self)
        # self.bn.draw_structure()
        edges = []
        for edge in self.bn.structure.edges:
            edges.append((edge[1], edge[0]))
        for edge in edges:
            self.bn.structure.add_edge(edge[0],edge[1])
            
        paths = list(nx.all_simple_paths(self.bn.structure, x, y))
        if len(paths) == 0:
            return True
        

        for edge in edges:
            self.bn.del_edge((edge[0],edge[1]))
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

    def summing_out(self, CPT: dict, variable: str):
        """
        function to sum out a variable from a cpt
        input:  - a cpt (dict)
                - the variable to maximize out from the cpt (str)
        output: the summed out cpt 
        """
        new_CPT = pd.DataFrame({})
        vars = [x for x in CPT if x not in ['p', 'maxed', variable]]

        # loop over every row in cpt
        for i in range(len(CPT)):
            # select instantiation from line i
            instantiation = pd.Series(dict(zip(vars,CPT.iloc[i][vars])))
            # get all compatible instantiations
            table = self.bn.get_compatible_instantiations_table(instantiation, CPT)
            # select row with highest p-value
            p_value = sum(table['p'])
            # add assignment of variable to row
            line = CPT.iloc[i].copy()
            line['p'] = p_value
            # add new summed out line to new cpt
            new_CPT = new_CPT.append(line, 
                        ignore_index=True)
            # delete summed out variable from cpt
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
            for pair in list(itertools.combinations(neighbors, 2)):
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
                for pair in list(itertools.combinations(neighbors, 2)):
                    if not interaction_graph.has_edge(pair[0], pair[1]):
                        new_edges[var] += 1 
            next_var = min(new_edges.keys(), key = lambda var: new_edges[var])
            pi.append(next_var)
            new_edges.pop(next_var)

        return pi
    

    def Variable_elimination(self, CPT, list):

        for variable in list:
            CPT = self.summing_out(CPT, variable)

        return CPT
    
####################################################################################################
# This part is from Xinhe

# first the prequisite functions:


    def d_sep(self, X: List[str], Y: List[str], Z: List[str]):

        bn_copy = copy.deepcopy(self.bn)
        still_pruning = True
        while still_pruning:
            still_pruning = False
            # delete leaf nodes
            for node in bn_copy.get_all_variables():
                if node not in X and node not in Y and node not in Z and len(bn_copy.get_children(node)) == 0:
                    bn_copy.del_var(node)
                    still_pruning = True

            # del outgoing edges
            for ev in Z:
                children = bn_copy.get_children(ev)
                for child in children:
                    edge = (ev, child)
                    bn_copy.del_edge(edge)
                    still_pruning = True

        # list of sets with connected parts of graph
        connections = list(nx.connected_components(nx.Graph(bn_copy.structure)))
        xy = set(X).union(set(Y))
        for con in connections:
            if con.issuperset(xy):
                print("not d-separated")
                return False
        print("d-separated")
        return True

        # given set of variables Q and evidence, prunes e &n
        # -> returns edge, node-pruned BN

    def prune(self, Q: Set[str], ev: pd.Series):

        if Q != {}:
            # node prune
            ev_nodes = set(ev.index)
            not_elim = ev_nodes.union(Q)
            # not_elim = Q.union(ev_nodes)
            elim = set(self.bn.get_all_variables()).difference(not_elim)
            for node in elim:
                if len(self.bn.get_children(node)) == 0:
                    self.bn.del_var(node)

        # edge-prune
        for node in list(ev.index):
            # self.bn.update_cpt(node, self.bn.get_compatible_instantiations_table(ev, self.bn.get_cpt(node)))
            for child in self.bn.get_children(node):
                edge = (node, child)
                self.bn.del_edge(edge)
                cr_d = {node: ev[node]}
                cr_ev = pd.Series(data=cr_d, index=[node])
                self.bn.update_cpt(child, self.bn.get_compatible_instantiations_table(cr_ev, self.bn.get_cpt(child)))
                self.bn.update_cpt(child, self.bn.get_cpt(child).drop(columns=[node]))

    def orderWidth(self, ordering: list) -> int:
        interaction_graph = self.bn.get_interaction_graph()
        w = 0

        for i in range(1, len(ordering)):
            d = len([k for k in interaction_graph.neighbors(ordering[i])])
            w = max(w, d)

            neighbors = interaction_graph.neighbors(ordering[i])
            interaction_graph.remove_node(ordering[i])

            for j in powerset(neighbors):
                if len(j) >= 2 and j[-1] is not None:
                    # print(j)  # for debugging
                    # print('created edges between', j, f'after removing {ordering[i]}')  # just for debugging.
                    interaction_graph.add_edge(j[0], j[1])

        return w

    def randomOrder(self, X: list) -> list:
        random.shuffle(X)
        return X

    def minDegreeOrder(self, X: list) -> list:
        interaction_graph = self.bn.get_interaction_graph()
        pi = list()
        nodes = dict()

        # counts all the edges to see which node has the least
        for i in X:
            for j in interaction_graph.neighbors(i):
                if j in nodes:
                    nodes[j] += 1
                else:
                    nodes[j] = 1


        for k in range(len(X)):
            var_least_edges = min(nodes, key=nodes.get)  # gets the var with the least edges
            pi.append(var_least_edges)  # puts it in the output list; lists are ordered so they're good for this
            nodes.pop(var_least_edges)  # removes this var from the counting dict

            # if there is only 1 neighbour, then just remove the node without creating any more edges
            if len(list(interaction_graph.neighbors(var_least_edges))) < 2:
                interaction_graph.remove_node(var_least_edges)
                continue

            # makes a list of neighbours
            neighbors = list(interaction_graph.neighbors(var_least_edges))
            # removes the node from the interaction graph
            interaction_graph.remove_node(var_least_edges)
            # iterates over the neighbours to create pairs and then create edges between the nodes
            for i in powerset(neighbors):
                if len(i) == 2 and not interaction_graph.has_edge(i[0], i[1]):
                    interaction_graph.add_edge(i[0], i[1])


        return pi

    def minFillOrder(self, X: list) -> list:
        interaction_graph = self.bn.get_interaction_graph()
        pi = list()
        nodes = dict()

        X_copy = X.copy()

        for m in range(len(X)):
            # calculate the score for every variable
            nodes = dict()
            for i in X_copy:
                # print(i, len(list(interaction_graph.neighbors(i))))
                degree = len(list(interaction_graph.neighbors(i)))
                y = 0
                for j in [j for j in itertools.combinations(list(interaction_graph.neighbors(i)), 2)]:
                    if not interaction_graph.has_edge(j[0], j[1]):
                        y += 1

                nodes[i] = (((degree - 1) * degree) / 2) - y
                # print(nodes)
                # print()

            min_edges_created = min(nodes, key=nodes.get)
            pi.append(min_edges_created)

            # removes the variable from the list and the dict
            nodes.pop(min_edges_created)
            X_copy.remove(min_edges_created)

            # makes a list of neighbours
            neighbors = list(interaction_graph.neighbors(min_edges_created))
            # removes the node from the interaction graph
            interaction_graph.remove_node(min_edges_created)
            # iterates over the neighbours to create pairs and then create edges between the nodes
            for k in powerset(neighbors):
                if len(k) == 2 and not interaction_graph.has_edge(k[0], k[1]):
                    # print(k)
                    # print('created edges between', k, f'after removing {min_edges_created}')
                    interaction_graph.add_edge(k[0], k[1])

        return pi

    def retrieve_cpt(self, variable):
        return self.bn.get_cpt(variable)

    def summing_out(self, CPT, index_same, list):
        # create the final new CPT without the variables that should be summed out
        new_CPT = CPT.copy()
        for variable in list:
            new_CPT = new_CPT.drop(columns=[variable])

        # loop through the keys from the dictionary
        for key in index_same:

            # pick the p_value from that key
            p_value_sum = CPT.iloc[key]['p']

            # pick the values from the key, so the equal indexes
            equal_indexes = [index_same.get(key)]

            # loop through the equal indexes
            for i in equal_indexes:
                # add this p-value from this row to the total p-val
                p = CPT.iloc[i]['p'].values
                p_value_sum += p[0]

                # set this value to zero to be able to delete it later
                new_CPT.at[i, 'p'] = 0

            # set the new_CPT key value to the new p value

            new_CPT.at[key, 'p'] = p_value_sum

        for index_CPT in range(len(new_CPT)):
            if new_CPT.iloc[index_CPT]['p'] == 0:
                new_CPT.drop([index_CPT])

        return new_CPT

    def check_double(self, CPT, list, type):
        """
        sum all the values from a list out

        CPT: pandas dataframe
        list: list

        returns: pandas dataframe
        """

        # create a dictionary for the equal rows with same values
        index_same = {}

        # create a CPT without p-values and without all the variables that should be summed out
        clean_CPT = CPT.copy()

        clean_CPT = clean_CPT.drop(columns=["p"])
        for variable in list:
            clean_CPT = clean_CPT.drop(columns=[variable])

        # loop trough the length of rows of the clean CPT
        for row_1 in clean_CPT.iloc:
            for i in range(row_1.name + 1, len(clean_CPT)):
                row_2 = clean_CPT.iloc[i]

                # compare the different rows
                if row_1.equals(row_2):

                    # if it is still empty just add the new key with index number and index of equal row as value
                    if row_1.name in index_same:
                        index_same[row_1.name].append(i)
                    else:
                        index_same[row_1.name] = [i]

        if type == "sum":
            new_CPT = self.summing_out(CPT, index_same, list)
        elif type == "max":
            new_CPT = self.maxing_out(CPT, index_same)

        return new_CPT

    def maxing_out(self, CPT, index_same):
        """
        # function to sum all the values from a list out
        """

        new_CPT = CPT.copy()

        # for variable in list:
        #     new_CPT = new_CPT.drop(columns=[variable])

        # loop through the keys from the dictionary
        for key in index_same:

            # pick the p_value from that key
            p_value_max = CPT.iloc[key]['p']

            # pick the values from the key, so the equal indexes
            equal_indexes = index_same.get(key)

            # loop through the equal indexes
            for i in equal_indexes:

                # add the max p-value from this row to the total p-val
                if CPT.iloc[i]['p'] > p_value_max:
                    p_value_max = CPT.iloc[i]['p']

                # set this value to zero to be able to delete it later
                new_CPT.at[i, 'p'] = 0

            # set the new_CPT key value to the new p value
            new_CPT.at[key, 'p'] = p_value_max

        for index_CPT in range(len(new_CPT)):
            if new_CPT.iloc[index_CPT]['p'] == 0:
                new_CPT.drop([index_CPT])

        return new_CPT

    def factor_mul_Xinhe(self, CPT_1, CPT_2):
        CPT_1 = CPT_1.reset_index(drop=True)
        CPT_2 = CPT_2.reset_index(drop=True)

        # matching columns
        columns_1 = list(CPT_1)
        columns_2 = list(CPT_2)
        columns = [x for x in columns_1 if x in columns_2]

        # create a dictionary for the equal rows with same values
        index_same = {}

        # create a CPT without p-values and without all the variables that should be summed out
        clean_CPT_1 = CPT_1.copy()
        clean_CPT_2 = CPT_2.copy()

        clean_CPT_1 = clean_CPT_1[columns]
        clean_CPT_1 = clean_CPT_1.drop(columns="p")

        clean_CPT_2 = clean_CPT_2[columns]
        clean_CPT_2 = clean_CPT_2.drop(columns="p")

        # loop trough the length of rows of the clean CPT
        for row_1 in clean_CPT_1.iloc:

            row_1 = row_1.replace([True], 1.0)
            row_1 = row_1.replace([False], 0.0)

            for row_2 in clean_CPT_2.iloc:
                row_2 = row_2.replace([True], 1.0)
                row_2 = row_2.replace([False], 0.0)

                # compare the different rows
                if row_1.equals(row_2):

                    # if it is still empty just add the new key with index number and index of equal row as value
                    if row_1.name in index_same:
                        index_same[row_1.name].append(row_2.name)
                    else:
                        index_same[row_1.name] = [row_2.name]

        new_columns = columns_1.copy()
        new_columns.remove('p')
        new_columns.extend(x for x in columns_2 if x not in new_columns)

        new_CPT = pd.DataFrame()
        merge_CPT_1 = CPT_1.copy()

        merge_CPT_2 = CPT_2.copy()

        for key, values in index_same.items():

            # merge rows
            row_1 = merge_CPT_1.iloc[key].drop("p")
            for value in values:
                # merge_CPT_2.reset_index()
                row_2 = merge_CPT_2.iloc[value].drop("p")

                difference = [name for name in new_columns if name not in columns_1]

                new_row = pd.merge(row_1, row_2[difference], left_index=True, right_index=True, how='outer')
                new_row = new_row.iloc[:, 0].fillna(new_row.iloc[:, 1])
                p_1 = CPT_1.iloc[key]["p"]
                p_2 = CPT_2.iloc[value]["p"]
                new_p = p_1 * p_2

                new_row["p"] = round(new_p, 8)

                new_CPT = new_CPT.append(new_row, ignore_index=True)

        return new_CPT

    # reduce_factor(self, CPT, list, 'sum') --> used for summing out
    # factor_mul(self, CPT_1, CPT_2) --> used for multiplying factor

    def cpt_mul(self, cpt1, cpt2):
        new_CPT = pd.DataFrame()
        columns2 = list(cpt2)
        columns1 = list(cpt1)
        # diff = [x for x in columns1 if x in columns2]
        # print(f'{diff} DIFFF')
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

# now the functions Xinhe was assigned:


    def marginal_distribution(self, Q, evidence, order):
        '''
        Q = variables in the network BN
        evidence = instantiation of some variables in the BN

        output = posterior marginal Pr(Q|e)
        '''

        self.prune(Q, evidence)
        not_Q = [x for x in self.bn.get_all_variables() if x not in Q]
        if order == 1:
            order = set(self.minDegreeOrder(not_Q))
        elif order == 2:
            order = set(self.minFillOrder(not_Q))
        elif order == 3:
            order = set(self.randomOrder(not_Q))

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

                factor = self.reduce_factor(final_cpt, [variable], 'sum')
                list_goed.append(factor)

            S = list_goed
        cpt_new=pd.DataFrame()
        for i in range(0, len(S) - 1):
            if len(set(list(S[i])).intersection(set(list(S[i])))) > 1:
                cpt_new = self.factor_mul(S[i], S[i + 1])
            else:
                cpt_new = self.cpt_mul(S[i], S[i + 1])
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
            order = set(self.minDegreeOrder(not_Q))
        elif order == 2:
            order = set(self.minFillOrder(not_Q))
        elif order == 3:
            order = set(self.randomOrder(not_Q))

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
                    cpt1 = self.cpt_mul(cpt1, cpt2)

                final_cpt = cpt1
                final_cpt = final_cpt[final_cpt['p'] != 0]

                if Q == {} or variable in Q:

                    cpt = self.reduce_factor(final_cpt, [variable], 'max')

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
                cpt_new = self.factor_mul_Xinhe(S[i], S[i + 1])

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



def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))
  

