# By Jackson Thissell
# 12/11/2023

import math
import random

import torch
from scipy.special import legendre, binom
from sklearn.model_selection import train_test_split

from torch.optim import Adam
from torch.nn import MSELoss

from functional_encoder import FunctionalAutoEncoder
from estimate_fractal_dimension import estimate_fractal_dimension

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameters
TRAIN_SIZE = 18432
TEST_SIZE = 18432

VEC_SIZE = 1
LEGENDRE_CAP = 2  # Req: LEGENDRE_CAP > VEC_SIZE

LATENT_SIZE = 3

LR = 0.0005
BATCH_SIZE = 64
EPOCHS = 64

DEVICE = 'cuda'

# Reporting paramters
REPORT_RATE = 4
PLOT_RESOLUTION = 50

# Generate dataset
l_gen = []
p_leg = [(i+1) for i in range(6)]
for i in range(VEC_SIZE + 4):
    arg = random.randrange(len(p_leg))
    pick = p_leg.pop(arg)
    l_gen.append(legendre(pick))

full_dataset = torch.Tensor([])
side_len = int(math.sqrt(TRAIN_SIZE + TEST_SIZE))
for i in range(TRAIN_SIZE + TEST_SIZE):
    x = -1 + 2 * (i % side_len) / side_len
    y = -1 + 2 * (i // side_len) / side_len
    Y = torch.Tensor([[legendre(3)(x) + legendre(5)(y)]])

    full_dataset = torch.cat((full_dataset, Y), 0)

train_set, test_set = train_test_split(full_dataset, test_size=1/2, shuffle=True)

# Determine layer size based on findings from and Kai Fong Ernest Chong
layer_size = int(binom(VEC_SIZE + LEGENDRE_CAP, LEGENDRE_CAP))
layer_amount = 2

print(f"Architecture -> v_sz: {VEC_SIZE}, lt_sz: {LATENT_SIZE}, ly_amt: {layer_amount}, ly_sz: {layer_size}.")

# Create autoencoder
FAE = FunctionalAutoEncoder(VEC_SIZE, LATENT_SIZE, layer_amount, layer_size).to(DEVICE)
loss_function = MSELoss()
optimizer = Adam(FAE.parameters(), lr=LR)

fractal_dimension = []
loss_history = []
for i in range(EPOCHS):
    train_loss = 0
    test_loss = 0

    train_perm = torch.randperm(train_set.size()[0])
    test_perm = torch.randperm(test_set.size()[0])

    FAE.train()
    for j in range(0, train_set.size()[0], BATCH_SIZE):

        # Get batch
        indices = train_perm[j:j+BATCH_SIZE]
        y = train_set[indices].to(DEVICE)

        # Train model
        optimizer.zero_grad()
        y_hat = FAE(y)

        loss = loss_function(y_hat, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        # Evaluate and plot on regular intervals
        if (j+BATCH_SIZE) % (TRAIN_SIZE/REPORT_RATE) == 0:
            FAE.eval()

            test_loss = 0
            latent_space = np.empty((0, 3), float)
            for q in range(0, test_set.size()[0], BATCH_SIZE):

                # Get batch
                indices = test_perm[q:q+BATCH_SIZE]
                y = test_set[indices].to(DEVICE)

                # Get latent space
                l = FAE.encode(y).numpy(force=True)

                latent_space = np.append(latent_space, l, axis=0)

                # Find loss
                y_hat = FAE(y)
                loss = loss_function(y_hat, y)
                test_loss += loss.item()

            loss_history.append(math.log(test_loss, 10))

            batch_num = int(j/BATCH_SIZE)
            full_batches = int(TRAIN_SIZE/BATCH_SIZE)

            latent_fdim = estimate_fractal_dimension(latent_space)
            fractal_dimension.append(latent_fdim)

            print(f"Batch {batch_num}/{full_batches} done. Train loss: {train_loss}. Test loss: {test_loss}. Fractal dim: {latent_fdim}")

            filename = 'plots/epoch-' + str(i) + "-batch-" + str(batch_num) + ".png"

            sample_latent = latent_space[np.random.randint(latent_space.shape[0], size=8192), :]
            x, y, z = sample_latent.T

            # Generate plots
            fig = plt.figure(figsize=(13, 11))
            ax21 = fig.add_subplot(221, projection='3d')
            ax21.scatter(x, y, z)
            ax21.set_title('Latent Space')

            ax3 = fig.add_subplot(223)
            ax3.plot(fractal_dimension)
            ax3.plot(loss_history)
            ax3.set_title("Fractal Dimension / Time")

            plot_actual_x = np.asarray([-1 + 2*i/PLOT_RESOLUTION for i in range(PLOT_RESOLUTION+1)])
            plot_actual_y = np.asarray([-1 + 2*i/PLOT_RESOLUTION for i in range(PLOT_RESOLUTION+1)])
            plot_actual_x, plot_actual_y = np.meshgrid(plot_actual_x, plot_actual_y)

            plot_actual_z = legendre(3)(plot_actual_x) + legendre(5)(plot_actual_y)
            tensor_z = torch.Tensor(plot_actual_z).cuda()

            mimic_tensor_z = FAE(tensor_z.reshape(-1, 1))
            plot_mimic_z = mimic_tensor_z.reshape(51, 51).numpy(force=True)

            ax = fig.add_subplot(222, projection='3d')
            ax.plot_wireframe(plot_actual_x, plot_actual_y, plot_actual_z,
                                          cmap=cm.coolwarm, linewidth=0, antialiased=False)

            ax.set_title("Actual Plot of Functions")

            ax2 = fig.add_subplot(224, projection='3d')
            ax2.plot_wireframe(plot_actual_x, plot_actual_y, plot_mimic_z,
                                          cmap=cm.coolwarm, linewidth=0, antialiased=False)

            ax2.set_title("Mimic Plot of Functions")

            plt.show()
            plt.savefig(filename)

            FAE.train()

            train_loss = 0
            test_loss = 0



    print(f"Epoch {i}/{EPOCHS} done.")

print('Training finished.')