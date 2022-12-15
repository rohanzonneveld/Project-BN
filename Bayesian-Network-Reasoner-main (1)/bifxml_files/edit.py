import random
import sys
import numpy as np
import os
from numpy.core.fromnumeric import var
import pandas as pd
from typing import Union, List, Tuple, Dict, Set
import itertools
from itertools import combinations
from pandas.core.frame import DataFrame
from BayesNet import BayesNet  
    
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
                order = set(self.minDegreeOrder(not_Q))
            elif order == 2:
                order = set(self.minFillOrder(not_Q))

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

                    factor = self.Variable_elimination(final_cpt, [variable], 'sum')
                    list_goed.append(factor)

                S = list_goed
            cpt_new=pd.DataFrame()
            for i in range(0, len(S) - 1):
                if len(set(list(S[i])).intersection(set(list(S[i])))) > 1:
                    cpt_new = self.factor_mul(S[i], S[i + 1])
                else:
                    cpt_new = self.cpt_mul(S[i], S[i + 1])
                    #TODO:need a function to multiply cpts

                S[i + 1] = cpt_new
            final_cpt = cpt_new
            final_cpt = final_cpt[final_cpt['p'] != 0]

            for var in list(cpt_new):
                if var != "p":
                    if var not in Q:
                        final_cpt = self.Variable_elimination(final_cpt, [var], 'sum')

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

                    cpt = self.Variable_elimination(final_cpt, [variable], 'max')

                else:
                    cpt = self.Variable_elimination(final_cpt, [variable], 'sum')

                cpt = cpt[cpt['p'] != 0]

                list_goed.append(cpt)

            S = list_goed

            # print(S)
        if len(S) == 1:
            cpt_new = S[0]
        else:
            for i in range(0, len(S) - 1):
                cpt_new = self.cpt_mul(S[i], S[i + 1])
                #TODO:need a function to multiply cpts

                S[i + 1] = cpt_new
        final_cpt = cpt_new

        highest_p = 0
        final_cpt = final_cpt[final_cpt['p'] != 0]

        length_final_cpt = len(final_cpt)
        for i in range(length_final_cpt):

            if final_cpt.iloc[i]["p"] > highest_p:
                highest_p = final_cpt.iloc[i]["p"]
                final_row = final_cpt.iloc[i]

        newest_cpt = pd.DataFrame()
        newest_cpt = newest_cpt.append(final_row, ignore_index=True)

        for column in newest_cpt:
            if column != "p":
                if column not in Q and Q != {}:
                    newest_cpt = newest_cpt.drop(columns=[column])
        
        # print(f"final_cpt {newest_cpt}")
        return newest_cpt