from BayesNet import BayesNet
from BNReasoner import BNReasoner
import copy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import timeit

testing_path=[ "testing/dog_problem.BIFXML",
               "testing/lecture_example.BIFXML",
               "testing/lecture_example2.BIFXML",
               "testing/task3.BIFXML"]

exp_path=[
            "bifxml_files/small_networks/lecture_example.BIFXML",
          "bifxml_files/small_networks/lecture_example2.BIFXML",
          "bifxml_files/small_networks/dog_problem.BIFXML",
          "bifxml_files/small_networks/asia.bifxml",
          "bifxml_files/small_networks/traffic.bifxml",
            "bifxml_files/small_networks/nodes15.bifxml",
          "bifxml_files/small_networks/nodes20.bifxml",
          "bifxml_files/small_networks/nodes21.bifxml",
          "bifxml_files/small_networks/nodes22.bifxml",
          "bifxml_files/small_networks/nodes23.bifxml",
            "bifxml_files/small_networks/nodes24.bifxml",

          ]

def test_prune():
    new_BN = BNReasoner(testing_path[1])
    # new_BN.bn.draw_structure()

    # new_BN.d_sep(["Sprinkler?"],["Slippery Road?"],["Rain?"])

    d = {'Winter?': True, 'Rain?': False}
    ev = pd.Series(data=d, index=['Winter?', 'Rain?'])

    query_vars = {"Wet Grass?"}
    new_BN.prune(query_vars, ev)

    new_BN.bn.draw_structure()
    print(new_BN.bn.get_cpt("Sprinkler?"))
    print("------------")
    print(new_BN.bn.get_cpt("Rain?"))
    print("------------")
    print(new_BN.bn.get_cpt("Wet Grass?"))

def plot():
    #running time MPE 312
    # random_rt = [0.2325157,0.2938359,0.25270,0.30720,2.46622]
    minFill_rt = [0.07215099599943642,0.08275698199940962,0.12477829700037546,0.24425570299990795,1.8550572180001836,12.1030728018]
    minDegree_rt = [0.07826822300012282,0.07409221699981572,0.11987377700006618,0.11133390299983148,1.478416572000242,8.0826021319]
    
    no_variables = [7,7,7,8,11,15]

    fig, ax = plt.subplots()
    # ax.plot(no_variables, random_rt, label="random")
    ax.plot(no_variables, minDegree_rt, label="minDegree")
    ax.plot(no_variables, minFill_rt, label="minFill")
    ax.legend()

    plt.show()

def main():



    BN = BNReasoner(exp_path[5])
    #BN.bn.draw_structure()

    # d = {'dysp': True }
    # evidence = pd.Series(data=d, index=['dysp'])

    start = timeit.default_timer()

    d = {'v4': False}
    evidence = pd.Series(data=d, index=['v4'])

    # print(BN.marginal_distribution({'light-on', 'dog-out', 'hear-bark'}, evidence, 3))
    
    
    
    
    # print(BN.marginal_distribution({'smoke'}, evidence))
    # print(BN.MAP_MPE({}, evidence, 2))
    stop = timeit.default_timer()

    print('Time: ', stop - start)
    plot()

if __name__ == '__main__':
    main()


