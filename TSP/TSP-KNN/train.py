from inc.tsp_args import *
from inc.tsp_core import *

args = args_init()
save_name = args.save_name
print(save_name)

net = Net(args).to(args.device)
net = net_train(args, net, verbose = True, save_name = save_name)