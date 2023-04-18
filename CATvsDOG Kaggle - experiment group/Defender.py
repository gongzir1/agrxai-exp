import numpy as np
import torch

# cos matrix 即是计算 两两余弦相似度
import FL_models


def cosine_distance_torch(x1, x2=None, eps=1e-8):   # 这里的x2默认没有输入，所以思路上跟欧氏距离那个代码一样
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)  # tensor.norm是取范数；方差归一化，即除以各自的模
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)  # tensor.t()是指转置tensor
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)  # 设定下限为min上限为max，超出范围直接截断为min或max值





class PoolingDef:
    def __init__(self, input_size:int, n_H: int, model, validation_X: torch.Tensor, validation_y: torch.Tensor, kernel=3):
        self.n_H = n_H  # 这个nh 指代是hidden layer的大小
        self.input_size = input_size
        self.stride = kernel
        self.kernel_size = kernel
        self.pool = torch.nn.MaxPool2d(self.kernel_size, self.stride)
        self.model = model
        self.validation_X = validation_X
        self.validation_y = validation_y


    def filter(self, grad: torch.Tensor, top_50_percent_param, replica_of_sum_grad, out_class,k=10, malicious_factor=0.2, pooling=True,
               normalize=True, mode="merge"):

        """ Change: only use that top 50% here;
            The grad in this case refers to the top50%filter that is actually used for comparison;
            while the replica is the full parameter gradient for 50 clients"""

        grad = top_50_percent_param # At this point the shape is for 50 people, top 50%


        if normalize:
            replica_of_sum_grad = torch.nn.functional.normalize(replica_of_sum_grad)
        replica = replica_of_sum_grad.clone()




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
                return replica[dense_selected]  # The indices go directly to the corresponding client itself


        if mode in ["merge", "cosine"]:
            cos_matrix = cosine_distance_torch(grad)
            k_nearest = torch.topk(cos_matrix, k=k, dim=1)
            neighbour_dist = torch.zeros(grad.size(0))
            for i in range(grad.size(0)):
                idx = k_nearest.indices[i]
                neighbour = cos_matrix[idx][:, idx]
                neighbour_dist[i] = neighbour.sum()
            cos_selected = torch.topk(neighbour_dist, k=selection_size).indices

            print(cos_selected)
            if mode == "cosine":
                return replica[cos_selected]

        if mode == "merge":
            union = torch.cat([dense_selected, cos_selected])
            uniques, count = union.unique(return_counts=True)
            selected = uniques[count>1]
        return replica[selected]


def fang_defense( malicious_factor: float, model, test_X: torch.Tensor, test_y: torch.Tensor,
                 n_H, input_size,
                 Fang_clients_last_10_param_replaced_by_0,
                 top_50_percent_param, replica_of_sum_grad,
                 pooling=False,
                 mode="combined", kernel=3, ):


    # replica is still self.sum_grad, which is the original param without the last50 set to 0.
    replica = replica_of_sum_grad.clone()

    # # This is the new data that has been replaced by last50 to 0 before,
    # and it is actually the data that is used to calculate the loss and acc for Fang
    grad = Fang_clients_last_10_param_replaced_by_0


    base_param = model.get_flatten_parameters()
    acc_rec = torch.zeros(grad.size(0))
    loss_rec = torch.zeros(grad.size(0))



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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):

        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []

        for target_layer in target_layers:

            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))


    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())


    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients


    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

def back_prop_clean_set_FLTrust(self_model, X: torch.Tensor, y: torch.Tensor, batch_size: int, local_epoch: int,
                        revert=False, FL_Trust_setting=False):

    param = self_model.get_flatten_parameters()

    loss = 0
    acc = 0


    my_activations_and_grads = ActivationsAndGradients(
        self_model.networks, reshape_transform=None, target_layers=[self_model.networks[1]])

    print([self_model.networks[1]])

    sum_Of_Round_weights_channel_wise = 0

    for epoch in range(local_epoch):
        batch_idx = 0
        while batch_idx * batch_size < X.size(
                0):
            lower = batch_idx * batch_size
            upper = lower + batch_size


            X_b = X[lower: upper]
            y_b = y[lower: upper]
            y_b = y_b.to(DEVICE)
            X_b = X_b.to(DEVICE)


            self_model.optimizer.zero_grad()


            out = my_activations_and_grads(X_b)
            out = out.to(DEVICE)



            print(y_b.shape)
            print(out.shape)

            # y_b = y_b.squeeze(dim=1)  #

            loss_b = self_model.loss(out, y_b)
            loss_b.backward()


            grads_list = [g.cpu().data.numpy() for g in
                          my_activations_and_grads.gradients]


            for layer_grads in zip(grads_list):


                layer_grads = layer_grads[
                    0]

                weights_channel_wise = self_model.get_cam_weights(
                    layer_grads)
                weights_channel_wise = weights_channel_wise.squeeze(-1)
                weights_channel_wise = weights_channel_wise.squeeze(-1)


                weights_channel_wise = np.sum(weights_channel_wise, axis=0)

                print(weights_channel_wise)

                sum_Of_Round_weights_channel_wise += weights_channel_wise

            self_model.optimizer.step()
            loss += loss_b.item()
            pred_y = torch.max(out, dim=1).indices
            acc += torch.sum(pred_y == y_b).item()


            batch_idx += 1




    weight_index = np.argsort(sum_Of_Round_weights_channel_wise)
    weight_index = weight_index[::-1]




    top_ten_weight_index = weight_index[0:10]
    last_ten_weight_index = weight_index[10:20]




    grad = self_model.get_flatten_parameters() - param

    loss /= local_epoch
    acc = acc / (local_epoch * X.size(0))

    my_activations_and_grads.release()


    if revert:

        self_model.load_parameters(param)

    return acc, loss, grad, top_ten_weight_index, last_ten_weight_index




def fl_trust(grad: torch.Tensor, validation_imgs: torch.Tensor, validation_label: torch.Tensor, model, batch_size,
            top_50_percent_param, replica_of_sum_grad,
             p_H_clients, local_epoch, n_H, output_size, pooling=False, kernel=3, ):

    """Here is the original 50 person grad param"""
    replica = replica_of_sum_grad.clone()  # grad_zero = {Tensor:(50,198660)}  # grad这个是self.sum_grad，也就是未知的，被用来和clean-dataset作比对的
    replica = replica.cuda()    # replica 即是 grad.clone()

    """Only the top 50% are compared here"""
    grad_top_50_percent_param = top_50_percent_param


    acc, loss, grad_zero, top_ten_weight_index, last_ten_weight_index = back_prop_clean_set_FLTrust(self_model=model, X=validation_imgs, y=validation_label,
                                                                                                               batch_size=batch_size, local_epoch=local_epoch,
                                                                                                    revert=True)     # grad_zero = {Tensor:(198660,)}

    grad_zero = grad_zero.to(DEVICE)

    """Go through the index of the top10 and get the param itself """


    current_AGR_param = torch.empty(0)
    current_AGR_param = current_AGR_param.to(DEVICE)
    for index_i in top_ten_weight_index:


        current_AGR_param = torch.hstack((current_AGR_param, grad_zero[280 + index_i * 91: 280 + index_i * 91 + 91]))  # current_client_param = {Tensor:(91,)}


    # Eventually, clients_top_10_param_matrix = {Tensor:(50,910)};
    # for each person recorded in the process each time is {Tensor:(1,910)}
    AGR_top_10_param_matrix = current_AGR_param
    AGR_top_10_param_matrix = AGR_top_10_param_matrix.to(DEVICE)  # GPU


    cos = torch.nn.CosineSimilarity(eps=1e-5)
    relu = torch.nn.ReLU()
    norm = grad_zero.norm()

    norm = norm.cuda()
    grad_top_50_percent_param = grad_top_50_percent_param.cuda()
    AGR_top_10_param_matrix = AGR_top_10_param_matrix.cuda()



    print(grad_top_50_percent_param.shape)
    print(AGR_top_10_param_matrix.shape)

    scores = cos(grad_top_50_percent_param, AGR_top_10_param_matrix)
    scores = relu(scores)


    fl_trust_selected_still_be_50clients = replica






    fl_trust_selected = torch.nn.functional.normalize(fl_trust_selected_still_be_50clients) * norm


    grad = (fl_trust_selected.transpose(0, 1) * scores).transpose(0, 1)  # 作用是交换矩阵的 指定的两个维度，transpose(dim0, dim1) → Tensor

    grad = torch.sum(grad, dim=0) / scores.sum()



    return grad



