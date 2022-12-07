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

BN = BNReasoner(testing_path[0])
cpts = BN.bn.get_all_cpts()
# print(cpts['dog-out'])
f1f2 = BN.factor_mul(dict(itertools.islice(cpts.items(), 2, 4)))
print(cpts['dog-out'])
print()
print(cpts['hear-bark'])
print()
print(f1f2)
