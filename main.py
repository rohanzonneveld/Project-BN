from BayesNet import BayesNet
from BNReasoner import BNReasoner

testing_path=[ "testing/dog_problem.BIFXML",
               "testing/lecture_example.BIFXML",
               "testing/lecture_example2.BIFXML",
               "testing/task3.BIFXML"]

exp_path=["bifxml_files/large_networks/win95pts.bifxml",
          "bifxml_files/medium_networks/medium.BIFXML",
          "bifxml_files/small_networks/asia.bifxml",
          "bifxml_files/small_networks/cancer.bifxml"]

BN = BNReasoner(testing_path[0])
cpt = BN.bn.get_all_cpts()['dog-out']
print(cpt)
new_cpt = BN.maxing_out(cpt,'family-out')
print(new_cpt)
second_cpt = BN.maxing_out(new_cpt, 'bowel-problem')
print(second_cpt)
