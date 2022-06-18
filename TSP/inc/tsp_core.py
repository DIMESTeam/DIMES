from inc.tsp_nets import *

def tsp_tune(emb0, phi_net, graph, opt_fn, steps, sample_size, greedy_size, verbose = True, plot = True, save_name = None):
    emb = emb0.detach().clone().requires_grad_()
    psi_net = phi_net.clone()
    psi_net.train()
    opt = opt_fn([emb, *psi_net.trainables()])
    tbar = range(1, steps + 1)
    if verbose:
        tbar = tqdm(tbar)
    y_means = []
    if greedy_size is not None:
        y_grs = [tsp_greedy(graph, par, greedy_size)[0].min().item()]
    else:
        y_grs = [None]
    for t in tbar:
        opt.zero_grad()
        par = psi_net(emb)
        ys, par_grad = tsp_softmax_grad(graph, par, sample_size, y_bl = None)
        y_means.append(ys.mean().item())
        if greedy_size is not None:
            y_grs.append(tsp_greedy(graph, par, greedy_size)[0].min().item())
        if verbose:
            tbar.set_description(f'step={t} y_mean={y_means[-1]:.4f} y_gr={y_grs[-1]}')
        #if not par_grad.isnan().any():
        par.backward(par_grad)
        opt.step()
        del par, ys, par_grad
    if plot:
        if save_name is None:
            save_name = f'tsp_tune~{time.time()}'
        if len(y_means) > 0:
            fig = plt.figure(figsize = (24, 12))
            ts = np.arange(1, len(y_means) + 1)
            sns.lineplot(x = ts, y = y_means)
            plt.title('y_mean vs t')
            fig.savefig(f'{save_name}~y_means.pdf')
            plt.show()
        if greedy_size is not None:
            fig = plt.figure(figsize = (24, 12))
            ts = np.arange(0, len(y_grs))
            sns.lineplot(x = ts, y = y_grs)
            plt.title('y_gr vs t')
            fig.savefig(f'{save_name}~y_grs.pdf')
            plt.show()
    return emb, psi_net, y_means, y_grs

def net_approx_grads(emb, psi_net, graph, sample_size):
    emb = emb.detach().clone().requires_grad_()
    if emb.grad is not None:
        emb.grad.zero_()
    par = psi_net(emb)
    _, par_grad = tsp_softmax_grad(graph, par.detach(), sample_size, y_bl = None)
    par.backward(par_grad)
    emb_grad = emb.grad.detach().clone()
    phi_grads = []
    for psi in psi_net.trainables():
        phi_grads.append(psi.grad.detach().clone())
    return emb_grad, phi_grads

def net_train(args, net, verbose = True, save_name = None):
    net.train()
    opt = args.outer_opt_fn(net.parameters())
    tbar = range(1, args.tr_outer_steps + 1)
    if verbose:
        tbar = tqdm(tbar)
        losses = []
    for step in tbar:
        graph_list, batch = Graph.gen_batch(args.tr_batch_size, args.device, args.n_nodes_min, args.n_nodes_max, args.knn_k)
        if verbose:
            y_list = []
        emb0_batch = net.emb_net(batch.x, batch.edge_index, batch.edge_attr)
        emb0_list = emb0_batch.split([graph.n_edges for graph in graph_list], dim = 0)
        emb0_grads = []
        phi_grad_lists = []
        for phi in net.par_net.trainables():
            phi_grad_lists.append([])
        for i, (graph, emb0) in enumerate(zip(graph_list, emb0_list)):
            if verbose:
                tbar.set_description(f'step {step}, graph {i + 1}')
            emb1, psi_net, ys, _ = tsp_tune(emb0, net.par_net, graph, args.inner_opt_fn, args.tr_inner_steps, args.tr_inner_sample_size, greedy_size = None, verbose = False, plot = False)
            if verbose:
                y_list.extend(ys)
            emb0_grad, phi_grads = net_approx_grads(emb1, psi_net, graph, sample_size = args.tr_inner_sample_size)
            emb0_grads.append(emb0_grad)
            for phi_grad_list, phi_grad in zip(phi_grad_lists, phi_grads):
                phi_grad_list.append(phi_grad)
        opt.zero_grad()
        emb0_batch.backward((torch.cat(emb0_grads, dim = 0) / args.tr_batch_size).detach()) # mean
        for phi, phi_grad_list in zip(net.par_net.trainables(), phi_grad_lists):
            torch_add_grad(phi, torch.stack(phi_grad_list, dim = 0).mean(dim = 0).detach())
        opt.step()
        del graph_list, batch, emb0_batch, emb0_list, emb0_grads, phi_grad_lists, phi, graph, emb0, emb1, psi_net, ys, _, emb0_grad, phi_grads, phi_grad_list
        gc.collect()
        if verbose:
            losses.append(np.mean(y_list))
            tbar.set_description(f'[step {step}] loss={losses[-1]:.4f}')
        net.eval()
        if save_name is not None:
            torch.save(net.state_dict(), f'{save_name}~net{step}.pt')
        net.train()
    if verbose:
        fig = plt.figure(figsize = (24, 12))
        ts = np.arange(1, len(losses) + 1)
        sns.lineplot(x = ts, y = losses)
        plt.title('loss vs step')
        if save_name is not None:
            fig.savefig(f'{save_name}~losses.pdf')
            np.savetxt(f'{save_name}~losses.txt', np.array(losses))
        plt.show()
    return net

def net_infer(args, net, graph, verbose = True, plot = True, save_name = None):
    with torch.no_grad():
        emb0 = net.emb_net(graph.x, graph.edge_index, graph.edge_attr)
    emb1, psi_net, _, _ = tsp_tune(emb0, net.par_net, graph, args.inner_opt_fn, args.te_tune_steps, sample_size = args.te_tune_sample_size, greedy_size = None, verbose = verbose, plot = plot, save_name = save_name)
    with torch.no_grad():
        psi_net.eval()
        par1 = psi_net(emb1.detach())
        y, tour = tsp_greedy(graph, par1, args.te_greedy_size)
        if verbose:
            print('y_gr:', y)
        if save_name is not None:
            torch.save(emb1, f'{save_name}~emb1.pt')
            torch.save(psi_net.state_dict(), f'{save_name}~psi_net.pt')
            torch.save(par1, f'{save_name}~par1.pt')
            torch.save(y, f'{save_name}~y.pt')
            torch.save(tour, f'{save_name}~tour.pt')
        return emb1, psi_net, par1, y, tour
