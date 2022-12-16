from BayesNet import BayesNet
from BNReasoner import BNReasoner
import pandas as pd
import time
import random
import tqdm
import csv

testing_path=[ "Projects/KR/Project-BN/testing/dog_problem.BIFXML",
               "Projects/KR/Project-BN/testing/lecture_example.BIFXML",
               "Projects/KR/Project-BN/testing/lecture_example2.BIFXML"
               ]

exp_path=["Projects/KR/Project-BN/bifxml_files/large_networks/win95pts.bifxml",
          "Projects/KR/Project-BN/bifxml_files/large_networks/b90-31.bifxml",
          "Projects/KR/Project-BN/bifxml_files/large_networks/b200-31.bifxml",
          "Projects/KR/Project-BN/bifxml_files/medium_networks/medium.BIFXML",
          "Projects/KR/Project-BN/bifxml_files/medium_networks/b30-101.bifxml",
          "Projects/KR/Project-BN/bifxml_files/small_networks/asia.bifxml",
          "Projects/KR/Project-BN/bifxml_files/small_networks/cancer.bifxml",
          "Projects/KR/Project-BN/bifxml_files/small_networks/lecture_example.BIFXML",
          "Projects/KR/Project-BN/bifxml_files/small_networks/lecture_example2.BIFXML",
          "C:/Users\Beheer/Python/Python39/Virtual Environments/.virtenv/Projects/KR/Project-BN/bifxml_files/small_networks/dog_problem.BIFXML",
          "Projects/KR/Project-BN/bifxml_files/traffic.bifxml"]

 
def pick_nodes(bn):
    nodes = list(bn.structure.nodes)
    random_nodes = random.sample(nodes,3)
    x,y,z = random_nodes[0], random_nodes[1], random_nodes[2]
    return x,y,z


def save_runtime(runtime: dict, filename: str):
    # Create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(runtime, orient = 'index')

    # Open the file in append mode
    with open(filename, 'a', newline='') as csvfile:
        # Create a CSV writer
        writer = csv.writer(csvfile)

        # Write the DataFrame to the CSV file
        df.to_csv(csvfile, index=False, header=False)

def main():
    string = 'Experiment_{}'
    count = 0
    runtime_dict = {}
    bn = BayesNet()
    bn.load_from_bifxml(exp_path[-1]) # THE ONLY THING YOU SHOULD CHANGE IS THE NUMBER IN EXP_PATH
    # bn.draw_structure()
    bnr = BNReasoner(bn)
    for i in tqdm.tqdm(range(10)):
        x,y,z = pick_nodes(bn)
    
        for j in tqdm.tqdm(range(3)):
            namestring = string.format(count)
            count +=1
            nodes, edges, sum = bnr.count_nodes_and_edges()
            start = time.time()
            dsep = bnr.d_separation(x,y,[z])
            stop = time.time()
            runtime = stop - start
            runtime = format(runtime, '.20f')
            runtime_dict.update({namestring: (nodes, edges, sum, runtime)})
    save_runtime(runtime_dict,'results_dsep.csv')      # UNCOMMENT THIS IF YOU WANT TO SAVE THE RESULTS
        # print(runtime_dict)
    

if __name__ == '__main__':
    main()
    
