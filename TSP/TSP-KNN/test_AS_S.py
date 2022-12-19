from inc.tsp_args import *
from inc.tsp_core import *

args = args_init()
save_name = tsp_save_name(args) if len(args.save_name) == 0 else args.save_name

# load net
net = Net(args).to(args.device)
net.load_state_dict(torch.load(f'{save_name}~net{args.te_net}.pt', map_location = args.device))

# load data
x_list = torch.tensor(np.load(f'../data/test-{args.n_nodes}-coords.npy'), dtype = torch.float32, device = args.device)
graph_list = [Graph.knn(x, args.knn_k) for x in tqdm(x_list[args.te_range_l : args.te_range_r])]

# test
y_list = []
for i in trange(args.te_range_l, args.te_range_r):
    save_name_i = f'{save_name}~graph{i}'
    graph = graph_list[i - args.te_range_l]
    _, _, _, y, _ = net_infer_sampling(args, net, graph, verbose = True, plot = True, save_name = save_name_i)
    y_list.append(y.item())
print(y_list)
print(np.mean(y_list))