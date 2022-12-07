from typing import Union
from BayesNet import BayesNet
import pandas as pd
import numpy as np
import sys
import itertools

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

    def maxing_out(self, CPT, variable):
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

    def factor_mul(self, CPTS):
        """
        function to perform factor multiplication between n factors
        Input: dict of CPTS
        Output: The CPT resulting from the factor multiplication
        """
        
        results = pd.DataFrame({})
        keys = list(CPTS.keys())
        # set first factor to multiply equal to first factor
        factor1 = CPTS[keys[0]]
        print(factor1)
        print()
        # loop over all other factors
        for factor in range(1,len(CPTS)+1):
            # set factor two equal to first not multiplied factor
            factor2 = CPTS[keys[factor]]
            print(factor2)
            print()
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
            factor1 = results
        
            CPT = factor1
            return CPT.drop_duplicates()

    def mindeg_order(self):
        pass

    def minfil_order(self):
        pass

