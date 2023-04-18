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

        '''target_layer.register_forward_hook() registers a "forward propagation" hook function 
            for each layer; when data flows through this layer, it is given to save_activation()'''
        for target_layer in target_layers:

            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    '''Note: Forward and backward propagation is in Inverse order, 
        so forward is .append() because forward propagation flows from the bottom to the top of the network'''

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    '''Note: Forward and backward propagation are in Inverse order, 
        so the reverse is used to plug into the top [0] position, 
        because the backward propagation flows from the top to the bottom of the network'''

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

        """Goes directly to a mean value in dimensions 2,3, which is, height and width;
        returns the weight of each channel relative to the activation"""
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        """Iterate through the images of this one batch, collecting all the losses"""
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    '''CAM core algorithms:
        weights return the weight of each channel relative to the activation; 
        as defined by CAM, multiply each activation by weight; do weighted summation'''

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)

        print(grads.shape)

        # Weighted first
        weighted_activations = weights * activations
        # Then sum up
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):

        """instantiate the class ActivationsAndGradients with self.activations in the instance,
        forward propagating the collected feature layer information"""
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]

        # Collecting gradient information
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]

        # Simple collection of image sizes
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            # get_cam_image() is the core algorithm that calculates the CAM for each layer
            cam = self.get_cam_image(layer_activations, layer_grads)
            # This is doing ReLU, discarding values less than 0
            cam[cam < 0] = 0
            # Image sizing with scale_cam_image()
            scaled = self.scale_cam_image(cam, target_size)
            # Add to the empty list and finally return all
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))

            # resize back to original image size
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # Forward propagation to get the network output logits,
        # where the __call__ method will be called to collect the output
        output = self.activations_and_grads(input_tensor)

        # CASE 1:
        # Check if it is int; it must be because it is the line number of the category which
        if isinstance(target_category, int):
            # input_tensor.size() returns the tensor shape, while input_tensor.size(0) is the first dimension,
            # the one added by .unsqueeze() before, which is the batch
            target_category = [target_category] * input_tensor.size(0)

        # CASE 2:
        # If target_category is not passed in,
        # CAM defaults to getting the category index with the highest network prediction score
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        # Emptying gradients
        self.model.zero_grad()

        loss = self.get_loss(output, target_category)
        # Then back propagate and trigger the previously registered hook function;
        # capture the gradient information and save
        loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor)

        return self.aggregate_multi_layers(cam_per_layer)

    # After execution jump to main grayscale_cam = grayscale_cam[0, :] This step

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


''' min_max - attack, the principle is to choose a direction, 
push the grad of a benign as far as possible in this direction, 
so that it is far from the real benign; 
this farthest distance is the maximum distance between the real benign and the two'''


def min_max(all_updates, model_re):
    deviation = torch.std(all_updates, 0)  # torch.std Calculate standard deviation
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


'''Scaling attack'''
def targeted_flip(train_img: torch.Tensor, target: int, backdoor_idx=7):
    augmented_data = train_img.clone()
    # modifying the colour depth of this batch of pixel points.
    augmented_data[:, backdoor_idx:backdoor_idx + 2] = 0.5
    # This is a batch of data that we have poisoned by setting all their labels to the category target.
    augmented_label = torch.ones(train_img.size(0), dtype=torch.long) * target
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

            ('conv1', torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)),  # 280 = { [(3*3) * 3] + 1 } * 10

            ('conv2', torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)),

            ('dropout1', nn.Dropout(0.5)),

            ('relu2', nn.ReLU()),
            ('Max2', torch.nn.MaxPool2d(kernel_size=3)),

            ('flatten', torch.nn.Flatten()),

            ('fc1', torch.nn.Linear(4500, 50)),

            ('relu3', nn.ReLU()),

            ('dropout1', nn.Dropout(0.5)),

            ('fc2', torch.nn.Linear(50, 2)),

        ]))

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = torch.nn.CrossEntropyLoss()
        self.grad = None

    def forward(self, x):

        print(x.shape)

        print(x.shape)
        return self.networks(x)

    def step(self):
        self.optimizer.step()

    @staticmethod
    def get_cam_weights(grads):
        # It is straightforward to find a mean value in dimensions 2,3, which is, height and width;
        # return the weight of each channel with respect to the activation
        return np.mean(grads, axis=(2, 3), keepdims=True)



    def back_prop(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, local_epoch: int, revert=False):

        param = self.get_flatten_parameters()

        loss = 0
        acc = 0


        for epoch in range(local_epoch):
            batch_idx = 0  # self.batch_size = self.train_imgs.size(0) // (self.Ph * self.batch)
            while batch_idx * batch_size < X.size(
                    0):
                lower = batch_idx * batch_size  # This is the start point of every (current) batch
                upper = lower + batch_size  # This is the end point of each (current) batch

                print(X.shape)

                print(lower)

                print(upper)

                X_b = X[lower: upper]  # Start-to-finish interception of the data of the current batch
                y_b = y[lower: upper]  # Start-to-finish interception of the label of the current batch
                y_b = y_b.to(DEVICE)
                X_b = X_b.to(DEVICE)



                self.optimizer.zero_grad()  # Zeroing the gradient first


                out = self.forward(X_b)

                out = out.to(DEVICE)

                loss_b = self.loss(out, y_b)
                loss_b.backward()


                self.optimizer.step()  # perform a step parameter (w, b) update by gradient descent (optimizer.step())
                loss += loss_b.item()  # Cumulatively collect the bosses and add them together
                pred_y = torch.max(out, dim=1).indices  # Compare the predicted label with the real and get the accuracy
                acc += torch.sum(pred_y == y_b).item()


                print(loss)
                batch_idx += 1  # Get the next  batch


        grad = self.get_flatten_parameters() - param
        loss /= local_epoch
        acc = acc / (local_epoch * X.size(0))


        if revert:
            self.load_parameters(param)  # If revert==True, then you have to restore and reload the param

        return acc, loss, grad



    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0)
        out = out.to(DEVICE)  # GPU

        with torch.no_grad():
            # .flatten expands from the ()th dimension, transforming the subsequent dimensions into one dimension.
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
        # # Press parameters (like parameters of global_model) in to self.parameters
        # (this is the iteration to get the parameters on local); and vice versa
        for param in self.parameters():
            with torch.no_grad():
                length = len(param.flatten())
                to_load = parameters[start_index: start_index + length]
                to_load = to_load.reshape(param.size())
                if mask is not None:  # Use mask only when sparsify, specify mask=idx
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

        """The clean set is cut out here and given to AGR to train the 50 mods that are uploaded."""

        self.agr_clean_imgs = None
        self.agr_clean_labels = None
        """This is the gradient given by the 50 clients in respond to for the clean set"""
        self.sum_grad_clean_set = None


        self.Ph = Ph  # Ph refers to the number of participant clients, set to be 50
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
        # # torch.unique() is to pick out the independent non-repeating elements of the tensor,
        # and by default it is sorted in ascending order
        self.out_class = torch.cat((torch.unique(self.test_labels), torch.unique(self.train_labels))).unique().size(
            0)
        self.global_model = ModelFC().to(DEVICE)  # GPU
        self.participants = []  # to store the clients
        self.loss = torch.nn.CrossEntropyLoss()

        self.sum_grad = None
        """Used to record the first 10 filters for each client"""
        self.sum_of_top_10_index = None

        self.malicious_index = None
        self.malicious_labels = None
        self.attack_mode = attack_mode
        self.scale_target = 0  # targeted class of the Scaling attack
        self.non_iid_labels = None
        self.non_iid_images = None
        self.non_iid_factor = non_iid_factor

    def federated_init(self):  # Used in the main program used to run, initialization in various _script

        # Here the GLOBAL parameters are taken and loaded onto the local model
        param = self.global_model.get_flatten_parameters()

        for i in range(self.Ph):
            model = ModelFC()

            model = model.to(DEVICE)  # GPU

            model.load_parameters(param)  # Initialize the client and load it onto each local model
            self.participants.append(model)  # Save each local-client to the list
        self.malicious_index = torch.zeros(self.Ph, dtype=torch.bool)  # Create malicious tags and create attacks
        # bernoulli_ generates Bernoulli distributions, generates toxic client locations at random
        self.malicious_index.bernoulli_(self.malicious_factor)

    def data_distribution(self, validation_size=2000, agr_clean_size=2000):
        self.batch_size = self.train_imgs.size(0) // (self.Ph * self.batch)

        print(self.batch_size)

        # Cut out the test-set and validation-set respectively
        self.validation_imgs = self.test_imgs[:validation_size]
        self.validation_labels = self.test_labels[:validation_size]

        self.agr_clean_imgs = self.test_imgs[validation_size: validation_size + agr_clean_size]
        self.agr_clean_labels = self.test_labels[validation_size: validation_size + agr_clean_size]

        self.test_imgs = self.test_imgs[validation_size + agr_clean_size:]
        self.test_labels = self.test_labels[validation_size + agr_clean_size:]



    def shuffle_data(self):
        shuffled_index = torch.randperm(self.train_imgs.size(0))
        self.train_imgs = self.train_imgs[shuffled_index]
        self.train_labels = self.train_labels[shuffled_index]
        shuffled_index = torch.randperm(self.train_imgs.size(0))
        self.malicious_labels = self.train_labels[shuffled_index]  # Disrupt and specify the label marked out as poisoned

        """In dataset 'CATvsDOG' need to shuffle test set too"""
        shuffled_index = torch.randperm(self.test_imgs.size(0))
        self.test_imgs = self.train_imgs[shuffled_index]
        self.test_labels = self.train_labels[shuffled_index]

    def get_training_data(self, idx: int, malicious=False):
        """This is the number of samples assigned to each participant,
        self.train_imgs is the set of images, and dimension 0 is how many there are of that"""
        sample_per_cap = self.train_imgs.size(0) // self.Ph

        low = idx * sample_per_cap
        high = low + sample_per_cap

        if self.non_iid_factor is None:  # CASE 01: 它直接是 iid的数据集
            # If the client in this round is malicious, then use the malicious label
            if malicious:
                return self.train_imgs[low: high], self.malicious_labels[low: high].flatten()
            # If the client in this round is not malicious, then use the normal label
            return self.train_imgs[low: high], self.train_labels[low: high].flatten()



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


        for i in range(self.Ph):  # Ph is the number of clients attending the study

            model = self.participants[i]  # Get the model for each local in the client list in turn
            if pipe_lost[i]:
                continue
            X, y = self.get_training_data(i)

            print(X.shape)

            torch.cuda.empty_cache()
            acc, loss, grad = model.back_prop(X, y, self.batch_size, self.local_epoch)  # This is the back_prop  of local

            self.collect_grad(i, grad)
            sum_acc += acc
            sum_loss += loss

        if attack and attack_mode == "min_max":  # CASE 01: "min_max" attack
            all_updates = self.sum_grad.clone()
            all_updates = all_updates[~self.malicious_index]
            for i in range(self.Ph):
                if not self.malicious_index[i]:  # If it is not a poisoned index, then this round will not be operated
                    continue
                local = self.sum_grad[i]
                mal_grad = min_max(all_updates, local)
                self.collect_grad(i, mal_grad)

        # CASE 02: other three attacks
        if attack and attack_mode in ["mislead", "grad_ascent"]:
            for i in range(self.Ph):

                # If it is not a poisoned index, then this round will not be operated
                if not self.malicious_index[i]:
                    continue
                model = self.participants[i]

                # attack_mode == "label_flip" is implemented here
                X, y = self.get_training_data(i, malicious=True)
                acc, loss, grad = model.back_prop(X, y, self.batch_size, self.local_epoch)
                local = self.sum_grad[i]

                grad = grad.cuda()
                local = local.cuda()

                # if attack_mode == "label_flip":
                #     mal_grad = grad  # Here is where to take the gradient from  above (already set malicious=True)
                if attack_mode == "grad_ascent":
                    mal_grad = - local
                else:  # that is == "mislead"
                    mal_grad = grad - local
                self.collect_grad(i, mal_grad)
        if attack and attack_mode in ["scale"]:  # CASE 03: scale attack
            for i in range(self.Ph):

                # If it is not a poisoned index, then this round will not be operated
                if not self.malicious_index[i]:
                    continue
                model = self.participants[i]
                X, y = self.get_training_data(i)
                X, y = targeted_flip(X, self.scale_target)  # A scale attack was performed
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
        return (sum_acc / self.Ph), (sum_loss / self.Ph)

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

        # Assign the value here, get the corresponding local_grad and append it to self.sum_grad[idx]
        self.sum_grad[idx] = local_grad


    def apply_grad(self):
        model = self.global_model

        # The model parameters are first obtained after the previous round of aggregation,
        # and the gradients of the sampled user model parameters are weighted and averaged.
        grad = torch.mean(self.sum_grad, dim=0)

        # Get the model parameters for the current round again
        param = model.get_flatten_parameters()
        grad = grad.cuda()

        # Then add to the model parameters after the previous round of aggregation to update the global model parameters
        param = param + grad
        model.load_parameters(param)

    def apply_pooling_def(self, clients_top_50_percent_param_matrix, replica_sum_grad):
        model = self.global_model
        defender = Defender.PoolingDef(self.train_imgs.size(1), self.n_H, model=model,
                                       validation_X=self.validation_imgs, validation_y=self.validation_labels,
                                       kernel=self.p_kernel)
        if self.defender in ["np-dense", "np-cosine", "np-merge"]:
            mode = self.defender[3:]


            grad = defender.filter(grad=self.sum_grad, out_class=self.out_class, k=self.k,
                                   malicious_factor=self.malicious_factor, pooling=False, mode=mode,
                                   top_50_percent_param=clients_top_50_percent_param_matrix,
                                   replica_of_sum_grad=replica_sum_grad)


        grad = torch.mean(grad, dim=0)  #
        self.last_grad = grad
        param = model.get_flatten_parameters()

        grad = grad.to(DEVICE)

        param = param + grad
        model.load_parameters(param)

    def apply_fang_def(self, Fang_clients_last_10_param_replaced_by_0, clients_top_50_percent_param_matrix,
                       replica_sum_grad, pooling=False, mode="combined"):
        model = self.global_model
        grad = Defender.fang_defense(replica_of_sum_grad=self.sum_grad, malicious_factor=self.malicious_factor,
                                     model=model, test_X=self.validation_imgs,
                                     test_y=self.validation_labels.flatten(), n_H=self.n_H, input_size=self.out_class,
                                     pooling=pooling, mode=mode, kernel=self.p_kernel,
                                     top_50_percent_param=clients_top_50_percent_param_matrix,
                                     Fang_clients_last_10_param_replaced_by_0=Fang_clients_last_10_param_replaced_by_0, )
        grad = torch.mean(grad, dim=0)
        param = model.get_flatten_parameters()

        grad = grad.to(DEVICE)

        param += grad
        model.load_parameters(param)

    def apply_fl_trust(self, replica_sum_grad, clients_top_50_percent_param_matrix, pooling=False):
        model = self.global_model
        grad = Defender.fl_trust(grad=self.sum_grad, validation_imgs=self.validation_imgs,
                                 validation_label=self.validation_labels.flatten(),
                                 model=model, batch_size=self.batch_size, local_epoch=self.local_epoch, n_H=self.n_H,
                                 output_size=self.out_class, pooling=pooling, kernel=self.p_kernel,
                                 replica_of_sum_grad=replica_sum_grad,
                                 top_50_percent_param=clients_top_50_percent_param_matrix, p_H_clients=self.Ph)
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


        print("我是self.test_imgs")
        print(self.test_imgs.shape)

        test_x, _ = targeted_flip(test_x, 0)
        test_y = self.test_labels.flatten()
        model = self.global_model

        print('我是test_x.shape')
        print(test_x.shape)

        print("我是test_x")
        print(test_x.shape)

        test_x = test_x.cuda()
        test_y = test_y.cuda()

        with torch.no_grad():
            out = model(test_x)

        loss_val = self.loss(out, test_y)  # self.scale_target is the class we tampered with, assumed to be class 0
        pred_y = torch.max(out, dim=1).indices  # test_y is the true y and pred_y is the predicted y given by the model
        idx = test_y != self.scale_target  # Condition 1. y of the sample itself is not the target class 0
        pred_y = pred_y[idx]
        test_y = test_y[idx]
        acc = torch.sum(pred_y == self.scale_target)  # Condition 2. model predicted by saying that this sample is the current target class 0
        acc = acc / test_y.size(0)
        return acc.item(), loss_val.item()

    def grad_sampling(self):
        sampling = torch.zeros(self.sum_grad.size(0), self.sum_grad.size(1) + 1)
        sampling[:, 0] = self.malicious_index
        sampling[:, 1:] = self.sum_grad
        nda = sampling.numpy()
        np.savez_compressed(self.output_path + f"grad_sample_{self.attack_mode}_{self.dataset}.npz", nda)


    """I took the clean backpass and wrote it into the FL_torch side, 
    making it possible to run it independent of the local"""
    # @staticmethod
    def back_prop_clean_set(self, self_model, X: torch.Tensor, y: torch.Tensor, batch_size: int, local_epoch: int,
                            revert=False, FL_Trust_setting=False):

        param = self_model.get_flatten_parameters()

        loss = 0
        acc = 0

        my_activations_and_grads = ActivationsAndGradients(
            self_model.networks, reshape_transform=None, target_layers=[self_model.networks[1]])

        print([self_model.networks[1]])

        """Here we record the weight of the last convolutional layer of the network with 20 filters (after 2 local epochs) 
        for 200 images at a time (assuming a total of 1000 local training images, after 2 local epochs are completed)"""
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

                # y_b = y_b.squeeze(dim=1)

                loss_b = self_model.loss(out, y_b)
                loss_b.backward()



                # Collecting gradient information
                grads_list = [g.cpu().data.numpy() for g in
                              my_activations_and_grads.gradients]


                for layer_grads in zip(grads_list):




                    layer_grads = layer_grads[0]

                    weights_channel_wise = self_model.get_cam_weights(layer_grads)
                    weights_channel_wise = weights_channel_wise.squeeze(-1)
                    weights_channel_wise = weights_channel_wise.squeeze(-1)

                    # Then sum by column, stacking 200 into 1 value
                    weights_channel_wise = np.sum(weights_channel_wise, axis=0)  # weights_channel_wise: ndarray(20,)

                    print(weights_channel_wise)

                    # Add to the global and finally go to the filter that contributes the most
                    sum_Of_Round_weights_channel_wise += weights_channel_wise

                self_model.optimizer.step()
                loss += loss_b.item()
                pred_y = torch.max(out, dim=1).indices
                acc += torch.sum(pred_y == y_b).item()


                batch_idx += 1  # 拿到下一个小批量的batch

                """Reload the param once after each batch for cleanset so that the current model is not affected"""
                if revert:
                    if FL_Trust_setting == True:
                        self_model.load_parameters(param)  # If revert==True, then have to restore and reload the param

        """1.0 """
        # Record the coordinates of each index (at this point x is still in disorder)
        weight_index = np.argsort(sum_Of_Round_weights_channel_wise)
        # index inversion for sorting from largest to smallest
        weight_index = weight_index[::-1]



        """2.0 Fetch the top10 weight - the corresponding filter"""
        # top_ten_sum_Of_Round_weights_channel_wise = sum_Of_Round_weights_channel_wise[0:10]
        # last_ten_sum_Of_Round_weights_channel_wise = sum_Of_Round_weights_channel_wise[10:20]
        top_ten_weight_index = weight_index[0:10]  # Top ten contributors to filter (top 50%)
        last_ten_weight_index = weight_index[10:20]  # The last ten contribute the least to the filter (the last 50%)



        """Normal updates"""
        grad = self_model.get_flatten_parameters() - param

        loss /= local_epoch
        acc = acc / (local_epoch * X.size(0))

        my_activations_and_grads.release()  # Always remember to release it


        """Note that for the sake of FL TRUst, the above revert has been commented out and reverted to here, 
        to be installed back in when running distance"""
        if revert:
            if FL_Trust_setting == True:
                self_model.load_parameters(param)  # If revert==True, then have to restore and reload the param

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
        for epoch in range(self.num_iter):  #
            self.collect_param()
            self.grad_reset()

            if epoch == self.start_attack:
                attacking = True
                print(f'Start attacking at round {epoch}')
            acc, loss = self.back_prop(attacking, self.attack_mode)

            """
            1.0 Here's where to start with clean set:
            Get the params of 50 people in turn and load them onto a temporary model"""

            replica = self.sum_grad.clone()


            current_model = ModelFC()
            current_model = current_model.to(DEVICE)  # GPU

            """Note: since the sum of all parameters for each filter in this network structure is 91, 
            and for the first 10 or last 10 filters recorded, the total number of parameters is 10*91 = 910.
            and it is these 10 filters for all clients that need to be recorded here, 
            so a tensor of the shape 50*910 needs to be constructed"""
            clients_top_10_param_matrix = torch.zeros(self.Ph, 910)
            clients_top_10_param_matrix = clients_top_10_param_matrix.to(DEVICE)  # GPU

            """Note: For Fang's mechanism, all the parameters of the model are needed 
            to validate the model to get the accuracy and loss, so here you need to construct a copy first, 
            setting the last 10 filters with the smallest contribution in each client to 0"""
            Fang_clients_last_10_param_replaced_by_0 = torch.zeros(self.Ph, 227252)  # 最终形状是{Tensor:(50,910)}
            Fang_clients_last_10_param_replaced_by_0 = Fang_clients_last_10_param_replaced_by_0.to(DEVICE)  # GPU 扔进去

            for client_i in range(self.Ph):  # 即是in range 50，依次拿到1-50每一个人
                client_grad = self.sum_grad[client_i]
                current_model.load_parameters(client_grad)

                """Here it should be: as in normal communication. 
                After distributing and loading the param, do a grad_reset_clean;
                But since we take the mods out of sum_grad and fill them each time, 
                the 'fill' process is a reset of all the grads (i.e. params) """


                """ Now use the model of the ith client, pass the clean set in to learn and record the top 10 index, 
                with the aim of getting the index of the filter with the larger and smaller contribution"""


                # self.agr_clean_labels = self.agr_clean_labels.squeeze(dim=1) # 在计算交叉熵时，y_b形状需要是一维，去掉那个1的维度，loss = loss(out,labels.squeeze(dim=1))

                batch_size_clean = 20

                """Note here I chose to revert once before the start of each batch, 
                this is to keep the current model, before the start of each batch is, 
                the original parameters in sum_grad"""

                """Note: clean's backprop writes in the FL_torch, on the AGR's side, 
                so it can be processed here without involving the local model at all"""
                acc_clean, loss_clean, grad_clean, top_10_index_clean, last_10_weight_index = \
                    self.back_prop_clean_set(current_model, self.agr_clean_imgs, self.agr_clean_labels,
                                             batch_size_clean, self.local_epoch, revert=True)

                """Go through the index of the top10 and get the param itself """
                current_client_param = torch.empty(0)  # Record each person's, save this 910 param
                for index_i in top_10_index_clean:  # Get the top10 index

                    # current_client_param = torch.hstack((current_client_param,
                    #                                      client_grad[280 + index_i * 91: 280 + index_i * 91 + 91]))  # current_client_param = {Tensor:(91,)}


                    """Note: 280 here means that since CAM uses filters in the last layer of the CNN, 
                    in order to get a specific filter in the last layer, 
                    since all parameters before the last layer are 280, you need to start at position 280 """
                    current_client_param = torch.hstack((current_client_param,
                                                         client_grad[
                                                         280 + index_i * 91: 280 + index_i * 91 + 91]))  # current_client_param = {Tensor:(91,)}

                """Eventually, clients_top_10_param_matrix = {Tensor:(50,910)}; 
                for each person recorded in the process each time is {Tensor:(1,910)}"""
                clients_top_10_param_matrix[ client_i] = current_client_param

                """Go through the index of last10 and get the param itself for Fang """
                current_client_param_last = torch.empty(0)  # 空的 记录每一个人的,存这910个param
                for index_i in last_10_weight_index:  # 拿到top10的index

                    # current_client_param = torch.hstack((current_client_param,
                    #                                      client_grad[280 + index_i * 91: 280 + index_i * 91 + 91]))  # current_client_param = {Tensor:(91,)}

                    """Here the last 10 filters with the smallest contribution are set to 0 for Fang's defence mechanism"""
                    current_client_param_last = client_grad  # current_client_param = {Tensor:(91,)}
                    current_client_param_last[280 + index_i * 91: 280 + index_i * 91 + 91] = 0

                """# Eventually, clients_top_10_param_matrix = {Tensor:(50,910)}; 
                for each person recorded in the process each time is {Tensor:(1,910)}"""
                Fang_clients_last_10_param_replaced_by_0[ client_i] = current_client_param_last



            if self.defender in ["p-dense", "p-cosine", "p-merge", "np-dense", "np-cosine", "np-merge"]:
                self.apply_pooling_def(clients_top_50_percent_param_matrix=clients_top_10_param_matrix,
                                       replica_sum_grad=replica)  # 1) Distance-based Mechanisms

            elif self.defender in ["fang", "lrr", "err", "p-fang"]:
                self.apply_fang_def(pooling=pooling, mode=self.defender,
                                    clients_top_50_percent_param_matrix=clients_top_10_param_matrix,
                                    Fang_clients_last_10_param_replaced_by_0=Fang_clients_last_10_param_replaced_by_0,
                                    replica_sum_grad=replica)  # 2) Prediction-based Mechanisms
            elif self.defender in ["fl_trust", "p-trust"]:
                self.apply_fl_trust(pooling=pooling, clients_top_50_percent_param_matrix=clients_top_10_param_matrix,
                                    replica_sum_grad=replica)  # 3) Trust Bootstrapping-based Mechanisms
            elif self.defender in ["tr_mean", "median", "p-tr"]:
                self.apply_other_def()  # 4)
            else:
                self.apply_grad()  #

            """Here, the attack and defence are completed, followed by the model evaluation"""

            if epoch % self.stride == 0:  # stride=10, That's every 10 epochs recorded
                if self.attack_mode == "scale":  # Targeted Attacks
                    test_acc, test_loss = self.evaluate_target()
                    print(
                        f'Epoch {epoch} - attack acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}'
                        f', train loss {loss:6.4f}')
                elif self.attack_mode == "label_flip":
                    test_acc, test_loss = self.evaluate_target()
                    print(f'Epoch {epoch} - attack asr {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}'
                          f', train loss {loss:6.4f}')
                else:  # Untargeted Attacks
                    test_acc, test_loss = self.evaluate_global()
                    print(f'Epoch {epoch} - test acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}'
                          f', train loss {loss:6.4f}')
                epoch_col.append(epoch)
                test_acc_col.append(test_acc)
                test_loss_col.append(test_loss)
                train_acc_col.append(acc)
                train_loss_col.append(loss)
        end_count = time.perf_counter()
        recorder = pd.DataFrame({"epoch": epoch_col, "test_asr": test_acc_col, "test_loss": test_loss_col,
                                 "train_acc": train_acc_col, "train_loss": train_loss_col})
        recorder.to_csv(
            self.output_path + f"{self.dataset}_Ph_{self.Ph}_nH_{self.n_H}_MF_{self.malicious_factor}_K_{self.p_kernel}_def_{self.defender}"
                               f"_attack_{self.attack_mode}_start_{self.start_attack}_" + str(self.non_iid_factor) +
            time_str + ".csv")
