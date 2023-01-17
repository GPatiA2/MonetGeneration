import sys, os

import torch

import Training
from StyleGANUPM import StyleDiscriminator
from StyleGANUPM.Components import make_dataset, make_logger
from StyleGANUPM.StyleGenerator import Generator

curentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(curentdir)
sys.path.append(parentdir)



output_dir = curentdir + "\\Models\\"
img_dir="C:/Users/Daniela/Documents/Datasets/monet_jpg"
logger = make_logger("project", output_dir, 'log')

def load(model, cpk_file):
    pretrained_dict = torch.load(curentdir + "\\Models\\" +  "\\models\\" + cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if __name__ == '__main__':

    loadingPrev = True
    generator_FILE = "GAN_GEN_5_40.pth"
    discriminator_FILE = "GAN_DIS_5_40.pth"
    generatorOptim_FILE = "GAN_GEN_OPTIM_5_40.pth"
    discriminatorOptim_FILE = "GAN_DIS_OPTIM_5_40.pth"
    genShadow = "GAN_GEN_SHADOW_5_40.pth"
    initialDepth = 5

    # Dataset
    dataset = make_dataset(resolution=256,
                            ##La resolucion la cargamos como si fuesen imagenes 128x128 para evitar problemas
                            folder=False,
                            img_dir=img_dir,
                            conditional=False)



    epochs = [1, 1, 1, 1, 100, 100, 50*12, 64*12, 64*12]

    batch_sizes = [64, 32, 16, 4, 4, 4, 2, 1, 1]

    trainer = Training.Style_Prog_Trainer(
                                        generator=Generator,
                                        discriminator=StyleDiscriminator.Discriminator,
                                        conditional=False,
                                         n_classes=2,
                                         resolution=256,
                                         num_channels=3,
                                         latent_size=512,
                                         loss= "logistic",
                                         drift=0.001,
                                         d_repeats=1,
                                         use_ema=True,
                                         ema_decay=0.999,
                                         device='cuda',
                                         checksave = False,
                                          load = False,
                                          load_dir= None,
                                          gen_load= None,
                                          disc_load = None,
                                          time_steps = True,
                                          time_epochs= True)

    # Resume training from checkpoints
    if loadingPrev:
        load(trainer.gen, generator_FILE)
        trainer.dis.load_state_dict(torch.load(curentdir + "\\Models\\"+  "\\models\\" + discriminator_FILE))
        load(trainer.gen_shadow, genShadow)
        trainer.gen_optim.load_state_dict(torch.load(curentdir + "\\Models\\" +  "\\models\\"+ generatorOptim_FILE))
        trainer.dis_optim.load_state_dict(torch.load(curentdir + "\\Models\\" +  "\\models\\"+ discriminatorOptim_FILE))

    trainer.train_for_epochs(dataset=dataset,
                    num_workers=1,
                    epochs=epochs,
                    batch_sizes=batch_sizes,
                    logger=logger,
                    output=output_dir,
                    num_samples=1,
                    start_depth=initialDepth,
                    feedback_factor=1,
                    checkpoint_factor=10)


    # # init the network
    # style_gan = StyleGAN(conditional=False,
    #                      n_classes=131,
    #                      resolution=128,
    #                      num_channels=3,
    #                      latent_size=512,
    #                      loss="logistic",
    #                      drift=0.001,
    #                      d_repeats=1,
    #                      use_ema=True,
    #                      ema_decay=0.999,
    #                      device='cuda')
    #
    # epochs = [4, 4, 4, 4, 8, 16, 32, 64, 64]
    #
    # batch_sizes = [8, 8, 8, 8, 8, 4, 2, 1, 1]
    #
    # # train the network
    # style_gan.train(dataset=dataset,
    #                 num_workers=1,
    #                 epochs=epochs,
    #                 batch_sizes=batch_sizes,
    #                 logger=logger,
    #                 output=output_dir,
    #                 num_samples=36,
    #                 start_depth=0,
    #                 feedback_factor=10,
    #                 checkpoint_factor=10)
