import math
from collections import OrderedDict

import torch
import pandas as pd
import numpy as np
import datetime, time

from PIL import Image
from torchvision.transforms import transforms

import Defender
import cv2
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H")




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
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))

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




def min_max(all_updates, model_re):
    deviation = torch.std(all_updates, 0)
    lamda = torch.Tensor([10.0]).float()
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    distance = torch.cdist(all_updates, all_updates)
    max_distance = torch.max(distance)
    del distance
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)
        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2
        lamda_fail = lamda_fail / 2
    mal_update = (model_re - lamda_succ * deviation)
    return mal_update


def targeted_flip(train_img: torch.Tensor, target: int, backdoor_idx=7):
    augmented_data = train_img.clone()
    augmented_data[:, backdoor_idx:backdoor_idx+2] = 0.5



    augmented_label = torch.ones(train_img.size(0), dtype=torch.long) * target
    return augmented_data, augmented_label

def targeted_label_flip(train_img: torch.Tensor, target: int, backdoor_idx=7):
    augmented_data = train_img.clone()
    # augmented_data[:, backdoor_idx:backdoor_idx+2] = 0.5
    augmented_label = torch.ones(train_img.size(0), dtype=torch.long) * target
    return augmented_label


class ModelFC(torch.nn.Module):
    def __init__(self, n_H: int, in_length=28 * 28, out_class=10):
        super().__init__()
        self.n_H = n_H

        self.networks = torch.nn.Sequential(OrderedDict([

            ('conv1', torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3 )),

            ('conv2', torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)),
            ('dropout1',nn.Dropout(0.5)),
            ('relu2', nn.ReLU()),
            ('Max2', torch.nn.MaxPool2d(kernel_size=2)),



            ('flatten', torch.nn.Flatten()),

            ('fc1', torch.nn.Linear(2880, 50)),

            ('relu3', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),

            ('fc3', torch.nn.Linear(50, 10)),

        ]))

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = torch.nn.CrossEntropyLoss()
        self.grad = None

    def forward(self, x):

        return self.networks(x)

    def step(self):
        self.optimizer.step()

    def back_prop(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, local_epoch: int, revert=False):
        param = self.get_flatten_parameters()
        loss = 0
        acc = 0

        # print('this is ixxxx')
        # print(X)
        for epoch in range(local_epoch):
            batch_idx = 0
            while batch_idx * batch_size < X.size(0):
                lower = batch_idx * batch_size
                upper = lower + batch_size

                X_b = X[lower: upper]
                y_b = y[lower: upper]


                y_b_batch = y_b.shape[0]

                # y_b = y_b.view(y_b_batch,1,1)
                y_b = y_b.to(DEVICE)        # 好错误：

                x_batch = X_b.shape[0]
                x_weight = int(math.sqrt(X_b.shape[1]))
                x_height = int(math.sqrt(X_b.shape[1]))



                X_b = X_b.to(DEVICE)


                self.optimizer.zero_grad()
                out = self.forward(X_b)
                out = out.to(DEVICE)



                # loss_b = self.loss(out, y_b.squeeze())
                loss_b = self.loss(out, y_b)
                loss_b.backward()





                self.optimizer.step()
                loss += loss_b.item()
                pred_y = torch.max(out, dim=1).indices
                acc += torch.sum(pred_y == y_b).item()


                batch_idx += 1

        grad = self.get_flatten_parameters() - param
        loss /= local_epoch
        acc = acc / (local_epoch * X.size(0))
        if revert:
            self.load_parameters(param)
        return acc, loss, grad

    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0)
        out = out.to(DEVICE)

        with torch.no_grad():
            for parameter in self.parameters():
                out = torch.cat([out, parameter.flatten()])
        return out

    def load_parameters(self, parameters: torch.Tensor, mask=None):
        """
        Load parameters to the current model using the given flatten parameters
        :param mask: only the masked value will be loaded
        :param parameters: The flatten parameter to load
        :return: None
        """
        start_index = 0
        for param in self.parameters():
            with torch.no_grad():
                length = len(param.flatten())
                to_load = parameters[start_index: start_index + length]
                to_load = to_load.reshape(param.size())
                if mask is not None:
                    local_mask = mask[start_index: start_index + length]
                    local_mask = local_mask.reshape(param.size())
                    param[local_mask] = to_load[local_mask]
                else:
                    param.copy_(to_load)
                start_index += length


class FL_torch:
    def __init__(self,
                 num_iter,
                 train_imgs,
                 train_labels,
                 test_imgs,
                 test_labels,
                 Ph,
                 malicious_factor,
                 defender,
                 n_H,
                 dataset,
                 batch=5,
                 sampling_prob=0.5,
                 max_grad_norm=1,
                 sigma=0,
                 start_attack=30,
                 attack_mode="min_max",
                 k_nearest=20,
                 p_kernel=3,
                 local_epoch=1,
                 stride=10,
                 pipe_loss=0,
                 output_path="./output/",
                 non_iid_factor=None):
        self.num_iter = num_iter
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.test_imgs = test_imgs
        self.test_labels = test_labels
        self.validation_imgs = None
        self.validation_labels = None
        self.Ph = Ph
        self.malicious_factor = malicious_factor
        self.defender = defender
        self.n_H = n_H
        self.batch = batch
        self.batch_size = 0
        self.dataset = dataset
        self.sampling_prob = sampling_prob
        self.max_grad_norm = max_grad_norm
        self.sigma = sigma
        self.start_attack = start_attack
        self.local_epoch = local_epoch
        self.stride = stride
        self.pipe_loss = pipe_loss
        self.output_path = output_path
        self.k = k_nearest
        self.p_kernel = p_kernel
        self.out_class = torch.cat((torch.unique(self.test_labels), torch.unique(self.train_labels))).unique().size(0)
        self.global_model = ModelFC(self.n_H, in_length=self.train_imgs.size(1), out_class=self.out_class).to(DEVICE)
        self.participants = []
        self.loss = torch.nn.CrossEntropyLoss()
        self.sum_grad = None
        self.malicious_index = None
        self.malicious_labels = None
        self.attack_mode = attack_mode
        self.scale_target = 0
        self.non_iid_labels = None
        self.non_iid_images = None
        self.non_iid_factor = non_iid_factor

    def federated_init(self):
        param = self.global_model.get_flatten_parameters()

        for i in range(self.Ph):

            model = ModelFC(self.n_H, in_length=self.train_imgs.size(1), out_class=self.out_class)


            model = model.to(DEVICE)

            model.load_parameters(param)
            self.participants.append(model)
        self.malicious_index = torch.zeros(self.Ph, dtype=torch.bool)
        self.malicious_index.bernoulli_(self.malicious_factor)

    def data_distribution(self, validation_size=300):
        self.batch_size = self.train_imgs.size(0) // (self.Ph * self.batch)
        print('我是self.batch_size')
        print(self.batch_size)

        self.validation_imgs = self.test_imgs[:validation_size]
        self.validation_labels = self.test_labels[:validation_size]
        self.test_imgs = self.test_imgs[validation_size:]
        self.test_labels = self.test_labels[validation_size:]



    def shuffle_data(self):
        shuffled_index = torch.randperm(self.train_imgs.size(0))
        self.train_imgs = self.train_imgs[shuffled_index]
        self.train_labels = self.train_labels[shuffled_index]
        shuffled_index = torch.randperm(self.train_imgs.size(0))
        self.malicious_labels = self.train_labels[shuffled_index]

    def get_training_data(self, idx: int, malicious=False):
        sample_per_cap = self.train_imgs.size(0) // self.Ph



        low = idx * sample_per_cap
        high = low + sample_per_cap

        if self.non_iid_factor is None:
            if malicious:
                return self.train_imgs[low: high], self.malicious_labels[low: high].flatten()
            return self.train_imgs[low: high], self.train_labels[low: high].flatten()

        else:
            extra_sample = self.non_iid_images.size(0) // self.Ph
            extra_low = idx * extra_sample
            extra_high = extra_low + extra_sample
            if malicious:
                return torch.vstack((self.train_imgs[low: high], self.non_iid_images[extra_low: extra_high])), \
                       self.malicious_labels[low+extra_low: high+extra_high].flatten()
            return torch.vstack((self.train_imgs[low:high], self.non_iid_images[extra_low:extra_high])), torch.cat((
                self.train_labels[low:high], self.non_iid_labels[extra_low: extra_high])).flatten()

    def grad_reset(self):
        if self.sum_grad is None:
            length = self.global_model.get_flatten_parameters().size(0)
            self.sum_grad = torch.zeros(self.Ph, length)
        else:
            self.sum_grad.zero_()

    def back_prop(self, attack=False, attack_mode="min_max"):
        sum_acc = 0
        sum_loss = 0
        pipe_lost = torch.zeros(self.Ph, dtype=torch.bool)
        pipe_lost.bernoulli_(p=self.pipe_loss)
        # print(pipe_lost)

        for i in range(self.Ph):

            model = self.participants[i]
            if pipe_lost[i]:
                continue
            X, y = self.get_training_data(i)






            torch.cuda.empty_cache()
            acc, loss, grad = model.back_prop(X, y, self.batch_size, self.local_epoch)

            self.collect_grad(i, grad)
            sum_acc += acc
            sum_loss += loss

        if attack and attack_mode == "min_max":
            all_updates = self.sum_grad.clone()
            all_updates = all_updates[~self.malicious_index]
            for i in range(self.Ph):
                if not self.malicious_index[i]:
                    continue
                local = self.sum_grad[i]
                mal_grad = min_max(all_updates, local)
                self.collect_grad(i, mal_grad)
        if attack and attack_mode in ["mislead", "grad_ascent"]:
            for i in range(self.Ph):
                if not self.malicious_index[i]:
                    continue
                model = self.participants[i]
                X, y = self.get_training_data(i, malicious=True)
                acc, loss, grad = model.back_prop(X, y, self.batch_size, self.local_epoch)
                local = self.sum_grad[i]

                grad = grad.cuda()
                local = local.cuda()


                if attack_mode == "grad_ascent":
                    mal_grad = - local
                else:
                    mal_grad = grad - local
                self.collect_grad(i, mal_grad)
        if attack and attack_mode in ["scale"]:
            for i in range(self.Ph):
                if not self.malicious_index[i]:
                    continue
                model = self.participants[i]
                X, y = self.get_training_data(i)
                X, y = targeted_flip(X, self.scale_target)
                acc, loss, grad = model.back_prop(X, y, self.batch_size, self.local_epoch)
                local = self.sum_grad[i]

                local = local.to(DEVICE)
                grad = grad.to(DEVICE)

                mal_grad = local + grad / self.malicious_factor
                self.collect_grad(i, mal_grad)

        if attack and attack_mode in["label_flip"]:
            for i in range(self.Ph):
                if not self.malicious_index[i]:
                    continue
                model = self.participants[i]
                X, y = self.get_training_data(i)
                y = targeted_label_flip(y, self.scale_target)

                acc, loss, grad = model.back_prop(X, y, self.batch_size, self.local_epoch)
                local = self.sum_grad[i]

                local = local.to(DEVICE)
                grad = grad.to(DEVICE)

                mal_grad = local + grad / self.malicious_factor
                self.collect_grad(i, mal_grad)
        return (sum_acc/self.Ph), (sum_loss/self.Ph)

    def collect_param(self, sparsify=False):
        param = self.global_model.get_flatten_parameters()
        pipe_lost = torch.zeros(self.Ph, dtype=torch.bool)
        pipe_lost.bernoulli_(p=self.pipe_loss)
        for i in range(self.Ph):
            if pipe_lost[i]:
                continue
            model = self.participants[i]
            if sparsify:
                to_load, idx = self.sparsify_update(param)
                model.load_parameters(to_load, mask=idx)
            else:
                model.load_parameters(param)

    def collect_grad(self, idx: int, local_grad: torch.Tensor, norm_clip=False, add_noise=False, sparsify=False):
        if norm_clip and local_grad.norm() > self.max_grad_norm:
            local_grad = local_grad * self.max_grad_norm / local_grad.norm()
        if add_noise:
            noise = torch.randn(local_grad.size()) * self.sigma
            if noise.norm() > self.max_grad_norm:
                noise = noise * self.max_grad_norm / noise.norm()
            local_grad = local_grad + noise
        if sparsify:
            local_grad, _ = self.sparsify_update(local_grad)

        self.sum_grad[idx] = local_grad

    def apply_grad(self):
        model = self.global_model
        grad = torch.mean(self.sum_grad, dim=0)
        param = model.get_flatten_parameters()

        grad = grad.cuda()
        param = param + grad
        model.load_parameters(param)

    def apply_pooling_def(self):
        model = self.global_model
        defender = Defender.PoolingDef(self.train_imgs.size(1), self.n_H, model=model,
                                       validation_X=self.validation_imgs, validation_y=self.validation_labels, kernel=self.p_kernel)
        if self.defender in ["np-dense", "np-cosine", "np-merge"]:
            mode = self.defender[3:]
            grad = defender.filter(grad=self.sum_grad, out_class=self.out_class, k=self.k,
                                   malicious_factor=self.malicious_factor, pooling=False, mode=mode)
        if self.defender in ["p-dense", "p-cosine", "p-merge"]:
            mode = self.defender[2:]
            grad = defender.filter(grad=self.sum_grad, out_class=self.out_class, k=self.k,
                                   malicious_factor=self.malicious_factor, pooling=True, mode=mode)
        grad = torch.mean(grad, dim=0)
        self.last_grad = grad
        param = model.get_flatten_parameters()

        grad = grad.to(DEVICE)

        param = param + grad
        model.load_parameters(param)

    def apply_fang_def(self, pooling=False, mode="combined"):
        model = self.global_model
        grad = Defender.fang_defense(self.sum_grad, self.malicious_factor, model, self.validation_imgs,
                                     self.validation_labels.flatten(), self.n_H, self.out_class, pooling, mode, kernel=self.p_kernel)
        grad = torch.mean(grad, dim=0)
        param = model.get_flatten_parameters()

        grad = grad.to(DEVICE)

        param += grad
        model.load_parameters(param)

    def apply_fl_trust(self, pooling=False):
        model = self.global_model
        grad = Defender.fl_trust(self.sum_grad, self.validation_imgs, self.validation_labels.flatten(),
                                 model, self.batch_size, self.local_epoch, self.n_H, self.out_class, pooling, kernel=self.p_kernel)
        param = model.get_flatten_parameters()
        param += grad
        model.load_parameters(param)

    def apply_other_def(self):
        if self.defender in ["tr_mean", "p-tr"]:
            grad = Defender.tr_mean(self.sum_grad, self.malicious_factor)
            grad = torch.mean(grad, dim=0)
        if self.defender == "median":
            grad = torch.median(self.sum_grad, dim=0).values
        model = self.global_model
        param = model.get_flatten_parameters()
        param = param + grad
        model.load_parameters(param)

    def sparsify_update(self, gradient, p=None):
        if p is None:
            p = self.sampling_prob
        sampling_idx = torch.zeros(gradient.size(), dtype=torch.bool)
        result = torch.zeros(gradient.size())
        sampling_idx.bernoulli_(p)
        result[sampling_idx] = gradient[sampling_idx]
        return result, sampling_idx

    def evaluate_global(self):
        test_x = self.test_imgs
        test_y = self.test_labels.flatten()
        model = self.global_model

        test_x = test_x.cuda()
        test_y = test_y.cuda()


        with torch.no_grad():
            out = model(test_x)

        out = out.cuda()

        loss_val = self.loss(out, test_y)
        pred_y = torch.max(out, dim=1).indices
        acc = torch.sum(pred_y == test_y)
        acc = acc / test_y.size(0)
        return acc.item(), loss_val.item()

    def evaluate_target(self):
        test_x = self.test_imgs

        test_x, _ = targeted_flip(test_x, 0)
        test_y = self.test_labels.flatten()
        model = self.global_model

        test_x = test_x.cuda()
        test_y = test_y.cuda()

        with torch.no_grad():
            out = model(test_x)

        loss_val = self.loss(out, test_y)
        pred_y = torch.max(out, dim=1).indices
        idx = test_y != self.scale_target
        pred_y = pred_y[idx]
        test_y = test_y[idx]
        acc = torch.sum(pred_y == self.scale_target)
        asr = acc / test_y.size(0)
        return asr.item(), loss_val.item()

    def grad_sampling(self):
        sampling = torch.zeros(self.sum_grad.size(0), self.sum_grad.size(1) + 1)
        sampling[:, 0] = self.malicious_index
        sampling[:, 1:] = self.sum_grad
        nda = sampling.numpy()
        np.savez_compressed(self.output_path+f"grad_sample_{self.attack_mode}_{self.dataset}_{self.local_epoch}", nda)

    def eq_train(self):
        epoch_col = []
        train_acc_col = []
        train_loss_col = []
        test_acc_col = []
        test_loss_col = []
        attacking = False
        pooling = False
        if self.defender.startswith("p"):
            pooling = True
        start_count = time.perf_counter()
        for epoch in range(self.num_iter):
            self.collect_param()
            self.grad_reset()
            if epoch == self.start_attack:
                attacking = True
                print(f'Start attacking at round {epoch}')
            acc, loss = self.back_prop(attacking, self.attack_mode)         # attack这里执行的
            if self.defender in ["p-dense", "p-cosine", "p-merge", "np-dense", "np-cosine", "np-merge"]:
                self.apply_pooling_def()                        # 1) Distance-based Mechanisms:
            elif self.defender in ["fang", "lrr", "err", "p-fang"]:
                self.apply_fang_def(pooling, self.defender)     # 2) Prediction-based Mechanisms:
            elif self.defender in ["fl_trust", "p-trust"]:
                self.apply_fl_trust(pooling)                    # 3) Trust Bootstrapping-based Mechanisms:
            elif self.defender in ["tr_mean", "median", "p-tr"]:
                self.apply_other_def()                          # 4)
            else:
                self.apply_grad()       #

            if epoch % self.stride == 0:
                if self.attack_mode == "scale":
                    test_acc, test_loss = self.evaluate_target()
                    print(f'Epoch {epoch} - attack asr {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}'
                          f', train loss {loss:6.4f}')
                elif self.attack_mode == "label_flip":
                    test_acc, test_loss = self.evaluate_target()
                    print(f'Epoch {epoch} - attack asr {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}'
                          f', train loss {loss:6.4f}')
                else:
                    test_acc, test_loss = self.evaluate_global()
                    print(f'Epoch {epoch} - test acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}'
                          f', train loss {loss:6.4f}')
                epoch_col.append(epoch)
                test_acc_col.append(test_acc)
                test_loss_col.append(test_loss)
                train_acc_col.append(acc)
                train_loss_col.append(loss)
        # end_count = time.perf_counter()
                if epoch == 70:
                    self.grad_sampling()
        # recorder = pd.DataFrame({"epoch": epoch_col, "test_acc": test_acc_col, "test_loss": test_loss_col,
        #                          "train_acc": train_acc_col, "train_loss": train_loss_col})
        # recorder.to_csv(
        #     self.output_path + f"{self.dataset}_Ph_{self.Ph}_nH_{self.n_H}_MF_{self.malicious_factor}_K_{self.p_kernel}_def_{self.defender}"
        #                        f"_attack_{self.attack_mode}_start_{self.start_attack}_" + str(self.non_iid_factor) +
        #     time_str +".csv")