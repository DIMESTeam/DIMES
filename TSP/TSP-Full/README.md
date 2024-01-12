# TSP-Full

If you want to test `TSP-Full`, you may modify the following function:

```py
def net_test(args, net, show = 0, verbose = False):
    net.eval()
    net.set_batch_size(args.te_batch_size)
    ys_nn = []
    for epoch in range(args.te_outer_epochs):
        x, adj = tsp_gen(args.te_batch_size, args.n_nodes)
        par0 = net(x, adj)
        par1 = tsp_optim(adj, par0, args.opt_inner_fn, args.te_inner_epochs, args.te_inner_samples, show = show if verbose and epoch == 0 else 0, verbose = verbose, x = x) # This is AS
        y_nn = tsp_greedy(adj, par1)[0] # You can replace tsp_greedy with tsp_sample
        ys_nn.append(y_nn)
    ys_nn = torch.cat(ys_nn, dim = 0)
    return ys_nn
```
