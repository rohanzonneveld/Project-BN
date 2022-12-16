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

def test_maxing_out(BN):
    cpt = BN.bn.get_all_cpts()['dog-out']
    print(cpt)
    new_cpt = BN.maxing_out(cpt,'family-out')
    print(new_cpt)
    second_cpt = BN.maxing_out(new_cpt, 'bowel-problem')
    print(second_cpt)

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



def test_d_sep():

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

def test_independence():
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

    print(BN.MAP_MPE({}, evidence, 2))


if __name__ == '__main__':
    # BN = BNReasoner(exp_path[-1])
    # BN.bn.draw_structure()


    # test_maxing_out(BN)
    # test_factor_mul()
    # test_mindeg_order(BN)
    # test_minfil_order(BN)
    # test_d_sep()
    # test_independence()
    # test_pruning()
    test_marginal_distribution()
    # test_MAP_MPE()
