from inc.tsp_args import *
from inc.tsp_core import *

# parse args

args = args_init()
save_name = tsp_save_name(args)

# train

net = Net(args).to(args.device)
net = net_train(args, net, verbose = True, save_name = save_name)

# infer

net.load_state_dict(torch.load(f'{save_name}~net{args.te_net}.pt', map_location = args.device))
x_list = torch.tensor(np.load(f'input/test-{args.n_nodes}-coords.npy'), dtype = torch.float32, device = args.device)
graph_list = [Graph.knn(x, args.knn_k) for x in tqdm(x_list[args.te_range_l : args.te_range_r])]
for i in trange(args.te_range_l, args.te_range_r):
    save_name_i = f'{save_name}~graph{i}'
    graph = graph_list[i - args.te_range_l]
    net_infer(args, net, graph, verbose = True, plot = True, save_name = save_name_i)
