import sys, os

from CycleGANUPM.Components import make_dataset

curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)



import CycleGenerator
import CycleDiscriminator
import Training
import torch
import torch.nn as nn
import Constants
from torch.utils.data import DataLoader

n_epochs = 200
batch_size = 2
lr = 0.00005
target_shape = 256
device = 'cuda'

# Modulos
gen_AB = CycleGenerator.Generator(3, 3).to(device)
gen_BA = CycleGenerator.Generator(3, 3).to(device)
gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=lr, betas=(0.5, 0.999))
disc_A = CycleDiscriminator.Discriminator(3).to(device)
disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
disc_B = CycleDiscriminator.Discriminator(3).to(device)
disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999))


from torchsummary import summary

summary(gen_AB, (3, 256, 256))
summary(disc_A, (3, 256, 256))


adv_criterion = nn.MSELoss() 
recon_criterion = nn.L1Loss() 

# Carpeta para resultados

training_dir = 'CycleTraining'
load_dir = 'models'
gen_disc_load = 'cycleGAN_256.pth'

img_dir1="C:/Users/Daniela/Documents/Datasets/monet_jpg"
img_dir2="C:/Users/Daniela/Documents/Datasets/photo_jpg"

# Dataset
dataset1 = make_dataset(resolution=256,
                       ##La resolucion la cargamos como si fuesen imagenes 128x128 para evitar problemas
                       folder=False,
                       img_dir=img_dir1,
                       conditional=False)

dataset2 = make_dataset(resolution=256,
                       ##La resolucion la cargamos como si fuesen imagenes 128x128 para evitar problemas
                       folder=False,
                       img_dir=img_dir2,
                       conditional=False)

dataLoader1 = DataLoader(dataset1, batch_size=Constants.BATCH_SIZE, shuffle=True)
dataLoader2 = DataLoader(dataset2, batch_size=Constants.BATCH_SIZE, shuffle=True)




criterion = nn.BCEWithLogitsLoss()
display_step = 1
checkpoint_step = 200

import gc

gc.collect()

torch.cuda.empty_cache()

trainer = Training.Cycle_Trainer(dataLoader1, dataLoader2, gen_AB, gen_BA, gen_opt, disc_A, disc_A_opt, disc_B, disc_B_opt, adv_criterion,
recon_criterion, display_step, training_dir, target_shape, 'cuda', True, checkpoint_step, True, load_dir, gen_disc_load, time_steps = True, time_epochs = True)


from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, FID
pytorch_fid_metric = FID(num_features=512, feature_extractor=trainer)

trainer.train_for_epochs(n_epochs)
