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

class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=True):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        # 就直接在2,3维度，也就是高和宽上，去求一个均值
        # 返回相对于activation每一个通道的权重
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        # 遍历这一个batch的图片，收集所有的loss
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    # CAM核心算法
    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        print("我是grad")
        print(grads.shape)
        # print(grads)
        # weights 返回相对于activation每一个通道的权重
        # 按CAM定义，即将每一个activation乘上weight；做加权求和
        # 先加权
        weighted_activations = weights * activations
        # 再求和
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        # 将类ActivationsAndGradients实例化后的instance中，的self.activations /正向传播收集到的特征层信息
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        # 收集梯度信息
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        # 简单收集图片大小
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            # get_cam_image() 就是核心算法，计算出每一个layer的CAM / 流程简单 直接点跳去看就完事
            cam = self.get_cam_image(layer_activations, layer_grads)
            # 这是在做ReLU, 舍弃小于0的数值
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            # scale_cam_image() 对图片尺寸处理
            scaled = self.scale_cam_image(cam, target_size)
            # 加入空列表中，最后全部返回
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer
    # 接下来跳到call函数的最后两行

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            # 先做个scale down
            # x = x - min(x)
            # x = x / max(x)
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            # 然后再resize回到原图尺寸
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        # 即cuda为Ture时，把input_tensor页转移到GPU上
        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        # 这里将调用42行的__call__方法，收集输出
        output = self.activations_and_grads(input_tensor)
        # 情况1：
        # 检查一下是否为int；必然是 因为这是category哪一类的行号
        if isinstance(target_category, int):
            # input_tensor.size()返回tensor形状，而input_tensor.size(0)就是第一个维度，也就是之前.unsqueeze()增加的那个维度，即batch
            # 即当前batch中图片数目 与 [target_category]元素个数是保持一致
            # [target_category] 列表中 作者原意是处理多个图片，不过咱main里面只用了一张图片
            target_category = [target_category] * input_tensor.size(0)

        # 情况2：
        # 如果咱不传入target_category，CAM默认去拿到 网络预测值分数最大的 类别索引
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        # 先清空一波梯度
        self.model.zero_grad()
        # get_loss() 看一下
        loss = self.get_loss(output, target_category)
        # 然后反向传播，并触发之前注册的hook函数；捕获梯度信息，并保存
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        #
        cam_per_layer = self.compute_cam_per_layer(input_tensor)

        # 最后将所有我们指定的 layer 的CAM进行融合(本例中只拿了最后1层，所以aggregate_multi_layers并不起啥作用)
        return self.aggregate_multi_layers(cam_per_layer)
    # 执行完后跳转到main中 grayscale_cam = grayscale_cam[0, :] 这一步

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    # 注意：通过applyColorMap()后的图片是 BGR格式，所以要用下一条指令转成RGB格式
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = np.float32(heatmap) / 255
    # 然后再缩放到0-1

    if np.max(img) > 1:     # 检查一下
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    # 这一步将两张图叠加一起

    # img = np.expand_dims(img, axis=2)  # 如果我手动 在2位置添加维数 *************

    # plt.imshow(img)
    # plt.show()
    #
    # plt.imshow(heatmap)
    # plt.show()

    cam = heatmap + img     # (28,28,3) + (28,28) 不太对
    cam = cam / np.max(cam)

    # plt.imshow(cam)
    # plt.show()

    """如果我直接用heatmap当做新的输入呢？"""
    # cam = heatmap
    return np.uint8(255 * cam)      # 除以最大值，乘上255，转成uinit8类型给回去


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img

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
    augmented_label = torch.ones(train_img.size(0), dtype=torch.long) * target  #
    return augmented_data, augmented_label
def targeted_label_flip(train_img: torch.Tensor, target: int, backdoor_idx=7):
    augmented_data = train_img.clone()
    # augmented_data[:, backdoor_idx:backdoor_idx+2] = 0.5
    augmented_label = torch.ones(train_img.size(0), dtype=torch.long) * target
    return augmented_label


class ModelFC(torch.nn.Module):
    def __init__(self, ):
        super().__init__()


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

            ('fc2', torch.nn.Linear(50, 10)),

        ]))

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = torch.nn.CrossEntropyLoss()
        self.grad = None



    def forward(self, x):

        return self.networks(x)

    def step(self):
        self.optimizer.step()

    @staticmethod
    def get_cam_weights(grads):

        return np.mean(grads, axis=(2, 3), keepdims=True)


    def back_prop(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, local_epoch: int, revert=False):

        param = self.get_flatten_parameters()

        loss = 0
        acc = 0



        for epoch in range(local_epoch):
            batch_idx = 0  # self.batch_size = self.train_imgs.size(0) // (self.Ph * self.batch)
            while batch_idx * batch_size < X.size(0):
                lower = batch_idx * batch_size
                upper = lower + batch_size


                X_b = X[lower: upper]
                y_b = y[lower: upper]
                y_b = y_b.to(DEVICE)
                X_b = X_b.to(DEVICE)



                self.optimizer.zero_grad()


                out = self.forward(X_b)

                out = out.to(DEVICE)


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

        self.agr_clean_imgs = None
        self.agr_clean_labels = None

        self.sum_grad_clean_set = None
        # self.current_model = ModelFC().to(DEVICE)


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
        self.global_model = ModelFC().to(DEVICE)   # GPU
        self.participants = []
        self.loss = torch.nn.CrossEntropyLoss()

        self.sum_grad = None

        self.sum_of_top_10_index = None

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

            model = ModelFC()


            model = model.to(DEVICE)

            model.load_parameters(param)
            self.participants.append(model)
        self.malicious_index = torch.zeros(self.Ph, dtype=torch.bool)       #
        self.malicious_index.bernoulli_(self.malicious_factor)

    def data_distribution(self, validation_size=2000, agr_clean_size = 2000):
        self.batch_size = self.train_imgs.size(0) // (self.Ph * self.batch)


        self.validation_imgs = self.test_imgs[:validation_size]
        self.validation_labels = self.test_labels[:validation_size]

        self.agr_clean_imgs = self.test_imgs[validation_size: validation_size + agr_clean_size]
        self.agr_clean_labels = self.test_labels[validation_size: validation_size + agr_clean_size]

        self.test_imgs = self.test_imgs[validation_size + agr_clean_size:]
        self.test_labels = self.test_labels[validation_size + agr_clean_size:]

    def non_iid_distribution(self, validation_size=300):
        if self.non_iid_factor is None:
            non_iid_factor=0.5
        else:
            non_iid_factor=self.non_iid_factor
        self.batch_size = self.train_imgs.size(0) // (self.Ph * self.batch)
        self.validation_imgs = self.test_imgs[:validation_size]
        self.validation_labels = self.test_labels[:validation_size]
        self.test_imgs = self.test_imgs[validation_size:]
        self.test_labels = self.test_labels[validation_size:]
        iid_length = round(self.train_imgs.size(0) * (1-non_iid_factor))
        non_iid_images = self.train_imgs[iid_length:]
        non_iid_labels = self.train_labels[iid_length:]
        self.train_imgs = self.train_imgs[:iid_length]
        self.train_labels = self.train_labels[:iid_length]
        idx = torch.sort(non_iid_labels.flatten()).indices
        self.non_iid_images = non_iid_images[idx]
        self.non_iid_labels = non_iid_labels[idx]

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

            length = self.global_model.get_flatten_parameters().size(0)
            self.sum_grad = torch.zeros(self.Ph, length)





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

        if attack and attack_mode == "min_max":     #
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


        self.sum_grad[idx] = local_grad


    def apply_grad(self):
        model = self.global_model
        grad = torch.mean(self.sum_grad, dim=0)
        param = model.get_flatten_parameters()

        grad = grad.cuda()
        param = param + grad
        model.load_parameters(param)

    def apply_pooling_def(self, clients_top_50_percent_param_matrix, replica_sum_grad):
        model = self.global_model
        defender = Defender.PoolingDef(self.train_imgs.size(1), self.n_H, model=model,
                                       validation_X=self.validation_imgs, validation_y=self.validation_labels, kernel=self.p_kernel)
        if self.defender in ["np-dense", "np-cosine", "np-merge"]:      #
            mode = self.defender[3:]


            grad = defender.filter(grad=self.sum_grad, out_class=self.out_class, k=self.k,
                                   malicious_factor=self.malicious_factor, pooling=False, mode=mode,
                                   top_50_percent_param=clients_top_50_percent_param_matrix, replica_of_sum_grad=replica_sum_grad)



        grad = torch.mean(grad, dim=0)
        self.last_grad = grad
        param = model.get_flatten_parameters()

        grad = grad.to(DEVICE)

        param = param + grad
        model.load_parameters(param)

    def apply_fang_def(self, Fang_clients_last_10_param_replaced_by_0, clients_top_50_percent_param_matrix, replica_sum_grad, pooling=False, mode="combined"):
        model = self.global_model
        grad = Defender.fang_defense(replica_of_sum_grad=self.sum_grad, malicious_factor=self.malicious_factor, model=model, test_X=self.validation_imgs,
                                     test_y=self.validation_labels.flatten(), n_H=self.n_H, input_size=self.out_class, pooling=pooling, mode=mode, kernel=self.p_kernel,
                                     top_50_percent_param=clients_top_50_percent_param_matrix,
                                     Fang_clients_last_10_param_replaced_by_0=Fang_clients_last_10_param_replaced_by_0,)
        grad = torch.mean(grad, dim=0)
        param = model.get_flatten_parameters()

        grad = grad.to(DEVICE)

        param += grad
        model.load_parameters(param)

    def apply_fl_trust(self, replica_sum_grad, clients_top_50_percent_param_matrix, pooling=False):
        model = self.global_model
        grad = Defender.fl_trust(grad=self.sum_grad, validation_imgs=self.validation_imgs, validation_label=self.validation_labels.flatten(),
                                 model=model, batch_size=self.batch_size, local_epoch=self.local_epoch, n_H=self.n_H, output_size=self.out_class, pooling=pooling, kernel=self.p_kernel,
                                 replica_of_sum_grad=replica_sum_grad, top_50_percent_param=clients_top_50_percent_param_matrix, p_H_clients=self.Ph)
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
        acc = acc / test_y.size(0)
        return acc.item(), loss_val.item()

    def grad_sampling(self):
        sampling = torch.zeros(self.sum_grad.size(0), self.sum_grad.size(1) + 1)
        sampling[:, 0] = self.malicious_index
        sampling[:, 1:] = self.sum_grad
        nda = sampling.numpy()
        np.savez_compressed(self.output_path+f"grad_sample_{self.attack_mode}_{self.dataset}.npz", nda)


    # @staticmethod
    def back_prop_clean_set(self, self_model, X: torch.Tensor, y: torch.Tensor, batch_size: int, local_epoch: int, revert=False, FL_Trust_setting = False):

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

                y_b = y_b.squeeze(dim=1)  #



                loss_b = self_model.loss(out, y_b)
                loss_b.backward()
                grads_list = [g.cpu().data.numpy() for g in
                              my_activations_and_grads.gradients]  # grads_list = {list: 1}


                for layer_grads in zip(grads_list):


                    layer_grads = layer_grads[0]

                    weights_channel_wise = self_model.get_cam_weights(layer_grads)
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


                if revert:
                    if FL_Trust_setting == True:
                        self_model.load_parameters(param)




        weight_index = np.argsort(sum_Of_Round_weights_channel_wise)
        weight_index = weight_index[::-1]



        top_ten_weight_index = weight_index[0:10]
        last_ten_weight_index = weight_index[10:20]





        grad = self_model.get_flatten_parameters() - param

        loss /= local_epoch
        acc = acc / (local_epoch * X.size(0))

        my_activations_and_grads.release()


        if revert:
            if FL_Trust_setting == True:
                self_model.load_parameters(param)

        return acc, loss, grad, top_ten_weight_index, last_ten_weight_index


    def cosine_distance_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


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
            acc, loss = self.back_prop(attacking, self.attack_mode)


            replica = self.sum_grad.clone()

            # self.current_model
            current_model = ModelFC()
            current_model = current_model.to(DEVICE)  #

            clients_top_10_param_matrix = torch.zeros(self.Ph, 910)
            clients_top_10_param_matrix = clients_top_10_param_matrix.to(DEVICE)  #

            Fang_clients_last_10_param_replaced_by_0 = torch.zeros(self.Ph, 146480)     #
            Fang_clients_last_10_param_replaced_by_0 = Fang_clients_last_10_param_replaced_by_0.to(DEVICE)  #

            for client_i in range(self.Ph):
                client_grad = self.sum_grad[client_i]
                current_model.load_parameters(client_grad)


                batch_size_clean = 20


                acc_clean, loss_clean, grad_clean, top_10_index_clean, last_10_weight_index = \
                    self.back_prop_clean_set(current_model, self.agr_clean_imgs, self.agr_clean_labels, batch_size_clean, self.local_epoch, revert=True)


                current_client_param = torch.empty(0)
                for index_i in top_10_index_clean:



                    current_client_param = torch.hstack((current_client_param,
                                                         client_grad[100 + index_i * 91: 100 + index_i * 91 + 91]))

                clients_top_10_param_matrix[client_i] = current_client_param


                current_client_param_last = torch.empty(0)
                for index_i in last_10_weight_index:



                    current_client_param_last = client_grad
                    current_client_param_last[100 + index_i * 91: 100 + index_i * 91 + 91] = 0

                Fang_clients_last_10_param_replaced_by_0[client_i] = current_client_param_last





            if self.defender in ["p-dense", "p-cosine", "p-merge", "np-dense", "np-cosine", "np-merge"]:
                self.apply_pooling_def(clients_top_50_percent_param_matrix=clients_top_10_param_matrix, replica_sum_grad=replica)

            elif self.defender in ["fang", "lrr", "err", "p-fang"]:
                self.apply_fang_def(pooling=pooling, mode=self.defender,
                                    clients_top_50_percent_param_matrix=clients_top_10_param_matrix, Fang_clients_last_10_param_replaced_by_0=Fang_clients_last_10_param_replaced_by_0,
                                    replica_sum_grad=replica)
            elif self.defender in ["fl_trust", "p-trust"]:
                self.apply_fl_trust(pooling=pooling,  clients_top_50_percent_param_matrix=clients_top_10_param_matrix, replica_sum_grad=replica)
            elif self.defender in ["tr_mean", "median", "p-tr"]:
                self.apply_other_def()
            else:
                self.apply_grad()

            if epoch % self.stride == 0:
                if self.attack_mode == "scale":
                    test_acc, test_loss = self.evaluate_target()
                    print(f'Epoch {epoch} - attack acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}'
                          f', train loss {loss:6.4f}')
                elif self.attack_mode == "label_flip":
                    test_acc, test_loss = self.evaluate_target()
                    print(f'Epoch {epoch} - attack acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}'
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
                if epoch == 70:
                    self.grad_sampling()
        # end_count = time.perf_counter()
        # recorder = pd.DataFrame({"epoch": epoch_col, "test_acc": test_acc_col, "test_loss": test_loss_col,
        #                          "train_acc": train_acc_col, "train_loss": train_loss_col})
        # recorder.to_csv(
        #     self.output_path + f"{self.dataset}_Ph_{self.Ph}_nH_{self.n_H}_MF_{self.malicious_factor}_K_{self.p_kernel}_def_{self.defender}"
        #                        f"_attack_{self.attack_mode}_start_{self.start_attack}_" + str(self.non_iid_factor) +
        #     time_str +".csv")