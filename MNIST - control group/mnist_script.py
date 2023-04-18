import numpy as np
from FL_models import *
import constants

data = np.load("./mnist_train.npz")
train_imgs = data['arr_0']
train_labels = data['arr_1']

data = np.load("./mnist_test.npz")
test_imgs = data['arr_0']
test_labels = data['arr_1']

train_imgs = torch.tensor(train_imgs, dtype=torch.float)
# train_imgs = train_imgs[0:6000, :]        # torch.Size([60000, 784])
# print('train_imgs.shape')
# print(train_imgs.shape)

test_imgs = torch.tensor(test_imgs, dtype=torch.float)
# test_imgs = test_imgs[0:1000, :]          # torch.Size([10000, 784])
# print('test_imgs.shape')
# print(test_imgs.shape)

train_labels = torch.tensor(train_labels, dtype=torch.long)
# train_labels = train_labels[0:6000, :]            # torch.Size([60000, 1])
# print('train_labels.shape')
# print(train_labels.shape)

test_labels = torch.tensor(test_labels, dtype=torch.long)
# test_labels = test_labels[0:1000, :]              # torch.Size([10000, 1])
# print('test_labels.shape')
# print(test_labels.shape)


print(f"Data loaded, training images: {train_imgs.size(0)}, testing images: {test_imgs.size(0)}")

print("Initializing...")
num_iter = 201
Ph = 50
hidden = 512
malicious_factor = 0.3




x_batch = train_imgs.shape[0]
train_imgs = train_imgs.view(x_batch, 1, 28, 28)


# print(train_imgs.shape)
torch.cuda.empty_cache()


x_batch = test_imgs.shape[0]
test_imgs = test_imgs.view(x_batch, 1, 28, 28)


# print(test_imgs.shape)
torch.cuda.empty_cache()



for att_mode in constants.all_the_attack:

    for exp in [
        # constants.baseline_dense,constants.baseline_fl_trust,constants.baseline_merge,constants.baseline_fang,constants.baseline_cos,
        #         constants.attacked,
                # constants.fang,
                # constants.fl_trust,
                constants.p_merge,
                # constants.np_dense, constants.np_cos
    ]:    # for exp in [constants.baseline, constants.attacked]:

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
            dataset="MNIST",
            start_attack=exp['start'],
            attack_mode=att_mode,
            k_nearest=35,
            p_kernel=2,
            local_epoch=2
        )
        cgd.shuffle_data()
        cgd.federated_init()
        cgd.grad_reset()
        cgd.data_distribution(2000)
        print(f"Start {att_mode} attack to {exp['defender']}...")

        print("我是train_imgs.shape")
        print(train_imgs.shape)
        # cgd.grad_sampling()
        # torch.cuda.empty_cache()
        cgd.eq_train()

        print(f"{att_mode} attack to {exp['defender']} complete")