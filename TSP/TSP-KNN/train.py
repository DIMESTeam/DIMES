from inc.tsp_args import *
from inc.tsp_core import *

args = args_init()
save_name = tsp_save_name(args) if len(args.save_name) == 0 else args.save_name

net = Net(args).to(args.device)
net = net_train(args, net, verbose = True, save_name = save_name)