from BayesNet import BayesNet
from BNReasoner import BNReasoner
import itertools
import os
import pandas as pd

testing_path=[ "testing/dog_problem.BIFXML",
               "testing/lecture_example.BIFXML",
               "testing/lecture_example2.BIFXML"
               ]

exp_path=["bifxml_files/large_networks/win95pts.bifxml",
          "bifxml_files/medium_networks/medium.BIFXML",
          "bifxml_files/small_networks/asia.bifxml",
          "bifxml_files/small_networks/cancer.bifxml",
          "bifxml_files/traffic.bifxml"]


def test_factor_mul():
    BN = BNReasoner('testing/dog_problem.BIFXML')
    cpts = BN.bn.get_all_cpts()
    f1f2 = BN.factor_mul(cpts['dog-out'], cpts['hear-bark'])
    print(cpts['dog-out'])
    print()
    print(cpts['hear-bark'])
    print()
    print(f1f2)

def test_maxing_out():
    BN = BNReasoner('testing/dog_problem.BIFXML')
    cpt = BN.bn.get_all_cpts()['dog-out']
    print(cpt)
    first_cpt = BN.maxing_out(cpt,'family-out')
    print(first_cpt)
    second_cpt = BN.maxing_out(first_cpt, 'bowel-problem')
    print(second_cpt)
    third_cpt = BN.maxing_out(second_cpt,'dog-out')
    print(third_cpt)

def test_mindeg_order(BN):
    X = BN.bn.get_all_variables()
    pi = BN.mindeg_order(X)
    print(f'pi = {pi}')

def test_minfil_order(BN):
    X = BN.bn.get_all_variables()
    pi = BN.minfil_order(X)
    print(f'pi = {pi}')

def create_usecase_structure():
    BN = BNReasoner(exp_path[-1])
    
    pos = {"Raining": (1, 10),
            "Daytime": (5, 10),
            "Weekend": (9, 10),
            "Busy": (0.5, 7.5),
            "Road-closure": (3, 7),
            "Public-transport-availability": (7, 7),
            "Public-transport-usage": (7.5, 4),
            "Traffic-congestion": (3.5, 4),
            "Accidents": (1, 2),
            "Delay": (6, 2),
            "Alternative-routes": (8.2, 1)
            }
    root = (46/256, 20/256, 0/256, 0.8)
    branch = (117/256, 51/256, 0/256, 0.8)
    leaf = (23/256, 144/256, 8/256, 0.93)
    node_color = [root, root, root, branch, branch, branch, leaf, branch, branch, leaf, leaf]

    BN.bn.draw_structure(pos=pos, node_color=node_color)

def test_pruning():
    bnr = BNReasoner("testing/lecture_example.BIFXML")
    bnr.bn.draw_structure()
    Query = 'Wet Grass'
    evidence = {'Winter?': True, 'Rain?': False}
    print(f"Query: {Query}, evidence: {evidence}")
    bnr.prune(['Wet Grass?'], {'Winter?': True, 'Rain?': False})
    bnr.bn.draw_structure()



def test_d_sep():

    # Test d-separation
    reasoner = BNReasoner('bifxml_files/small_networks/cancer.bifxml')
    # load the Bayesian network and create the BNReasoner object

    x = 'Smoker'
    y = 'Cancer'
    z = ['Pollution']

    if reasoner.d_separation(x, y, z):
        print(f'{x} and {y} are d-seperated given {z}')
    else:
        print(f'{x} and {y} are not d-separated given {z}')

def test_independence():
    bnr = BNReasoner("testing/lecture_example.BIFXML")
    x = 'Winter?'
    y = 'Wet Grass?'
    z = 'Sprinkler?'

    independent = bnr.is_independent(x, y, [z])
    
    if independent:
        print(f'{x} and {y} are independent given {z}')
    else:
        print(f'{x} and {y} are not independent given {z}')


def test_marginal_distribution():
    file = os.path.join("testing", "dog_problem.BIFXML")
    
    # test case
    BN = BNReasoner(file)

    d = {'family-out': False}
    evidence = pd.Series(data=d, index=['family-out'])

    print(BN.marginal_distribution({'light-on', 'dog-out', 'hear-bark'}, evidence, 2))

def test_MAP_MPE():
    file = os.path.join("testing", "dog_problem.BIFXML")
    
    # test case
    BN = BNReasoner(file)

    d = {'family-out': False}
    evidence = pd.Series(data=d, index=['family-out'])
    Q = BN.MAP_MPE(BN.bn.get_all_variables())
    print(Q)
    # print(Q, evidence, 2))

def test_summing_out():
    BN = BNReasoner('testing/dog_problem.BIFXML')
    cpt = BN.bn.get_all_cpts()['dog-out']
    print(cpt)
    first_cpt = BN.summing_out(cpt,'family-out')
    print(first_cpt)
    
def test_variable_elim():
    BN = BNReasoner('testing/dog_problem.BIFXML')
    cpt = BN.bn.get_all_cpts()['dog-out']
    print(cpt)
    vars = ['dog-out', 'bowel-problem', 'family-out']
    result = BN.Variable_elimination(cpt, vars)
    print(result)


if __name__ == '__main__':
    BN = BNReasoner(exp_path[-1])
    # BN.bn.draw_structure()

    # test_pruning() 
    # test_d_sep()
    # test_independence()
    # test_summing_out()
    # test_maxing_out()
    # test_factor_mul()
    # test_mindeg_order(BN)
    # test_minfil_order(BN)
    # test_variable_elim() 
    # test_marginal_distribution() #TODO
    # test_MAP_MPE() #TODO
