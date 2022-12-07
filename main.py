from BayesNet import BayesNet
from BNReasoner import BNReasoner
import itertools

testing_path=[ "testing/dog_problem.BIFXML",
               "testing/lecture_example.BIFXML",
               "testing/lecture_example2.BIFXML",
               "testing/task3.BIFXML"]

exp_path=["bifxml_files/large_networks/win95pts.bifxml",
          "bifxml_files/medium_networks/medium.BIFXML",
          "bifxml_files/small_networks/asia.bifxml",
          "bifxml_files/small_networks/cancer.bifxml"]


def test_factor_mul(BN):
    
    cpts = BN.bn.get_all_cpts()
    f1f2 = BN.factor_mul(dict(itertools.islice(cpts.items(), 2, 4)))
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


if __name__ == '__main__':
    BN = BNReasoner(testing_path[0])
    # BN.bn.draw_structure()

    # test_maxing_out(BN)
    # test_factor_mul(BN)
    # test_mindeg_order(BN)
    # test_minfil_order(BN)