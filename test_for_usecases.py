from BayesNet import BayesNet
from BNReasoner import BNReasoner



BN = BayesNet()
BN.load_from_bifxml("bifxml_files/traffic.bifxml")
reasoner = BNReasoner(BN)

Q = ['traffic-congestion']
evidence = {'Accidents': True}
print(reasoner.marginal_distribution(Q,evidence,2))
