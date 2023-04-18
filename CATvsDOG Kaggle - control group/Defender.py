import torch


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)



class PoolingDef:
    def __init__(self, input_size:int, n_H: int, model, validation_X: torch.Tensor, validation_y: torch.Tensor, kernel=3):
        self.n_H = n_H
        self.input_size = input_size
        self.stride = kernel
        self.kernel_size = kernel
        self.pool = torch.nn.MaxPool2d(self.kernel_size, self.stride)
        self.model = model
        self.validation_X = validation_X
        self.validation_y = validation_y

    def filter(self, grad: torch.Tensor, out_class,k=10, malicious_factor=0.2, pooling=True,
               normalize=True, mode="merge"):
        if normalize:
            grad = torch.nn.functional.normalize(grad)
        replica = grad.clone()



        replica = grad.clone()




        selection_size = int(round(grad.size(0) * (1 - malicious_factor)))
        selected = torch.zeros(grad.size(0), dtype=torch.bool)

        if mode in ["merge", "dense"]:
            dist_matrix = torch.cdist(grad, grad)
            k_nearest = torch.topk(dist_matrix, k=k, largest=False, dim=1)
            neighbour_dist = torch.zeros(grad.size(0))
            for i in range(grad.size(0)):
                idx = k_nearest.indices[i]
                neighbour = dist_matrix[idx][:, idx]
                neighbour_dist[i] = neighbour.sum()
            dense_selected = torch.topk(neighbour_dist, largest=False, k=selection_size).indices
            if mode == "dense":
                return replica[dense_selected]

        if mode in ["merge", "cosine"]:
            cos_matrix = cosine_distance_torch(grad)
            k_nearest = torch.topk(cos_matrix, k=k, dim=1)
            neighbour_dist = torch.zeros(grad.size(0))
            for i in range(grad.size(0)):
                idx = k_nearest.indices[i]
                neighbour = cos_matrix[idx][:, idx]
                neighbour_dist[i] = neighbour.sum()
            cos_selected = torch.topk(neighbour_dist, k=selection_size).indices
            print('我是cos_selected')
            print(cos_selected)
            if mode == "cosine":
                return replica[cos_selected]

        if mode == "merge":
            union = torch.cat([dense_selected, cos_selected])
            uniques, count = union.unique(return_counts=True)
            selected = uniques[count>1]
        return replica[selected]


def fang_defense(grad: torch.Tensor, malicious_factor: float, model, test_X: torch.Tensor, test_y: torch.Tensor,
                 n_H, input_size, pooling=False,
                 mode="combined", kernel=3):
    base_param = model.get_flatten_parameters()
    acc_rec = torch.zeros(grad.size(0))
    loss_rec = torch.zeros(grad.size(0))
    replica = grad.clone()



    base_param = base_param.cuda()

    for i in range(grad.size(0)):
        local_grad = grad[i]

        local_grad = local_grad.cuda()

        param = base_param + local_grad
        model.load_parameters(param)
        acc, loss, g = model.back_prop(X=test_X, y=test_y, batch_size=test_X.size(0), local_epoch=1)
        acc_rec[i] = acc
        loss_rec[i] = loss
    model.load_parameters(base_param)
    k_selection = int(round(grad.size(0) * (1 - malicious_factor)))
    ERR = torch.topk(acc_rec, k_selection).indices
    if mode == "err":
        return replica[ERR]
    LRR = torch.topk(loss_rec, k_selection, largest=False).indices
    if mode == "lrr":
        return replica[LRR]
    union = torch.cat([ERR, LRR])
    uniques, counts = union.unique(return_counts=True)
    final_idx = uniques[counts > 1]
    return replica[final_idx]


def tr_mean(grad: torch.Tensor, malicious_factor: float):
    m_count = int(round(grad.size(0) * malicious_factor))
    sorted_grad = torch.sort(grad, dim=0)[0]
    return sorted_grad[m_count: -m_count]


def fl_trust(grad: torch.Tensor, validation_imgs: torch.Tensor, validation_label: torch.Tensor, model, batch_size,
             local_epoch, n_H, output_size, pooling=False, kernel=3):
    replica = grad.clone()
    acc, loss, grad_zero = model.back_prop(validation_imgs, validation_label, batch_size, local_epoch, revert=True)
    grad_zero = grad_zero.unsqueeze(0)






    cos = torch.nn.CosineSimilarity(eps=1e-5)
    relu = torch.nn.ReLU()
    norm = grad_zero.norm()

    norm = norm.cuda()
    grad = grad.cuda()
    grad_zero = grad_zero.cuda()
    replica = replica.cuda()

    scores = cos(grad, grad_zero)
    scores = relu(scores)
    grad = torch.nn.functional.normalize(replica) * norm
    grad = (grad.transpose(0, 1) * scores).transpose(0, 1)
    grad = torch.sum(grad, dim=0) / scores.sum()
    return grad



