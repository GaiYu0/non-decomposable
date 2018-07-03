import copy
import torch as th
import my


def guided_es(module, eval, p, std, alpha):
    numel = sum(x.data.numel() for x in module.parameters())
    device = next(module.parameters()).device
    n = std * (alpha / numel) ** 0.5 * th.randn(p, numel, device=device)
    k = std * (1 - alpha) ** 0.5 * th.randn(p, 1, device=device)
    n_list, k_list = th.chunk(n, p, 0), th.chunk(k, p, 0)

    plus, minus = copy.deepcopy(module), copy.deepcopy(module)
    p_module, p_plus, p_minus = module.parameters, plus.parameters, minus.parameters

    delta_list = []
    numel_list = [x.numel() for x in p_module()]
    grad = th.cat([x.grad.view(1, -1) for x in p_module()], 1)

    for epsilon_n, epsilon_k in zip(n_list, k_list):
        epsilon = epsilon_n + epsilon_k * grad

        i = 0
        for numel, x_plus, x_minus in zip(numel_list, p_plus(), p_minus()):
            x_plus.data += epsilon[0, i : i + numel].view(x_plus.data.shape)
            x_minus.data -= epsilon[0, i : i + numel].view(x_minus.data.shape)
            i += numel

        delta_list.append(eval(minus) - eval(plus))

        for x_module, x_plus, x_minus in zip(p_module(), p_plus(), p_minus()):
            x_plus.data[:], x_minus.data[:] = x_module.data, x_module.data

    delta = th.tensor(delta_list, device=device).unsqueeze(1)
    g = 1 / (2 * std ** 2) * th.mean(delta * (n + k * grad), 0)

    i = 0
    for numel, x in zip(numel_list, p_module()):
        x.grad[:] = g[i : i + numel].view(x.grad.shape)
        i += numel

    '''
    new_grad = th.cat([x.grad.view(1, -1) for x in p_module()], 1)
    print(th.norm(grad) / th.norm(new_grad))
    print(th.nn.CosineSimilarity()(grad, new_grad))
    '''
