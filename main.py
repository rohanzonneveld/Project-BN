import BayesNet
import BNReasoner
import pandas as pd
import random

# functions: compute runtime, save runtimes in dict, convert to pd.dataframe and convert to csv: save as 'prune_runtime.csv' Joost
# For runtime we can use time.time before and after the program ran and subtract those

def save_runtime(runtime: dict, filename: str):
    df = pd.DataFrame.from_dict(runtime, orient = 'index')
    df.to_csv(filename)



# get networks from website and convert to BIFXML Xinhe
# We already have 7, get 3 more from website in whatsapp

# for network create 5 querys Joost
# Simplest thing might be to have a list with all the files of the networks in BIFXML format
for filename in networks_list:
    bn = BayesNet()
    bn.load_from_bifxml(filename)
    bnr = BNReasoner(bn)
    nodes = list(bn.structure.nodes)
    random_nodes = random.sample(nodes,3)
    x,y,z = random_nodes[0], random_nodes[1], random_nodes[2]
    bnr.d_separation(str(x),str(y),[z])
# Prune vs unprune(query, network) Joost

# Min deg vs min fill (query, network) Xinhe