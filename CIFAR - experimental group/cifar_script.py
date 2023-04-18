import numpy as np
from FL_models import *
import constants

train_imgs = torch.load("CIFAR_train_image.pth",map_location="cuda:0")


test_imgs = torch.load("CIFAR_test_image.pth",map_location="cuda:0")


train_labels = torch.load("CIFAR_train_labels.pth",map_location="cuda:0")


test_labels = torch.load("CIFAR_test_labels.pth",map_location="cuda:0")





print(f"Data loaded, training images: {train_imgs.size(0)}, testing images: {test_imgs.size(0)}")
print("Initializing...")
num_iter = 201
Ph = 50
hidden = 1024
malicious_factor = 0.3
for att_mode in constants.all_the_attack:
    for exp in [
                # constants.baseline,
                # constants.baseline_dense,constants.baseline_fl_trust,constants.baseline_merge,constants.baseline_fang,constants.baseline_cos,
                # constants.attacked,
                constants.fl_trust,
                constants.fang,
                constants.np_merge, constants.np_dense,
                constants.np_cos,
                ]:

        cgd = FL_torch(
            num_iter=num_iter,
            train_imgs=train_imgs,
            train_labels=train_labels,
            test_imgs=test_imgs,
            test_labels=test_labels,
            Ph=Ph,
            malicious_factor=malicious_factor,
            defender=exp['defender'],
            n_H=hidden,
            dataset="CIFAR",
            start_attack=exp['start'],
            attack_mode=att_mode,
            k_nearest=35,
            p_kernel=2,
            local_epoch=2
        )
        cgd.shuffle_data()
        cgd.federated_init()
        cgd.grad_reset()
        cgd.data_distribution(validation_size=2000, agr_clean_size=1000)
        print(f"Start {att_mode} attack to {exp['defender']}...")
        cgd.eq_train()
        print(f"{att_mode} attack to {exp['defender']} complete")