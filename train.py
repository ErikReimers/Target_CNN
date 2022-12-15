import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from model import get_model, PSNR
from generator import trainingDataGenerator, valDataGenerator, PlotGenerators

#This class decays the learning rate according the the % of epoches that have passed
class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125

#Get the inputs from the user
def get_args():
    parser = argparse.ArgumentParser(description='train noise2noise model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_noise_dir_list', type=str,nargs='*', required=True,
                        help='training noisy volumes dir')
    parser.add_argument('--train_target_dir_list', type=str,nargs='*', required=True,
                        help='training noisy volumes dir')
    parser.add_argument('--val_noise_dir_list', type=str, nargs='*',required=True,
                        help='validation noisy volumes dir')
    parser.add_argument('--val_target_dir_list', type=str, nargs='*',required=True,
                        help='validation noisy volumes dir')
    parser.add_argument('--output_path', type=str, default='checkpoints',
                        help='checkpoint dir')
    parser.add_argument('--volume_shape', nargs='*', type=int, default=[128,128,89],
                        help='What is the shape of the PET volumes, z y x')
    parser.add_argument('--model', type=str, default='unet',
                        help='model architecture (srresnet or unet)')
    parser.add_argument('--nb_epochs', type=int, default=5,
                        help='number of epochs')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='How often (epoch #s) do you save the weights?')
    parser.add_argument('--steps', type=int, default=10,
                        help='steps per epoch')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--loss', type=str, default='mse',
                        help='loss; mse, or mae, is expected')
    parser.add_argument('--weight', type=str, default=None,
                        help='weight file for restart')
    parser.add_argument('--slices',type=int,default=3,
                        help='how many slices to include for 2.5D, odd numbers only')
    parser.add_argument('--image_size', type=int, default=128,
                        help='training patch size')
    parser.add_argument('--nb_val', type=int, default=300,
                        help='how many validation images to include')
    parser.add_argument('--train_nb_augmentations', type=int, default=0,
                        help='How many times do you want to augment the data')
    parser.add_argument('--val_nb_augmentations', type=int, default=0,
                        help='How many times do you want to augment the data')
    parser.add_argument('--degrees_max',type=float,default=10,
                        help='How many degrees (+/- degrees) do you want to rotate the data by?')
    parser.add_argument('--distance_max', type=int, default=3,
                        help='How many pixels (+/- pixels) do you want to translate the data by?')
    parser.add_argument('--crop_xy', type=int, default=0,
                        help='Would you like to crop_xy the image to a size?, 0 will not crop_xy')
    parser.add_argument('--crop_z', type=int, default=0,
                        help='Would you like to cropz the image to a size?, 0 will not crop_z')
    parser.add_argument('--offset_z', type=int, default=0,
                        help='Would you like to offset the image in the z direction?')
    parser.add_argument('--padding', type=str, default='zero',
                        help='If you want to adding padding in the z direction to the volumes')
    parser.add_argument('--disp_examples', type=str, default='False',
                        help='display example patches before starting')
    parser.add_argument('--noise_norm_constant', type=float, default=1,
                        help='Normalization constant for the noisy data')
    parser.add_argument('--target_norm_constant', type=float, default=1,
                        help='Normalization constant for the target data')
    parser.add_argument('--nb_to_remove', type=int, default=0,
                        help='Number of snaps to remove from the beginning of the data')
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    #If a 0 is given for cropping, assume no cropping and use the original image size
    if args.crop_xy == 0:
        args.crop_xy = args.volume_shape[0]
    if args.crop_z == 0:
        args.crop_z = args.volume_shape[2]

    #Output_path
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)

    # Get the model from the model.py file
    model = get_model(model_name=args.model, slices=args.slices)

    #If the user specified inital weights then load those in as a starting point
    if args.weight is not None:
        model.load_weights(args.weight)

    opt = Adam(learning_rate=args.lr)
    callbacks = []

    #Compile the model and get the image pair generator classes
    model.compile(optimizer=opt, loss=args.loss, metrics=[PSNR])

    t_gen = trainingDataGenerator(noise_dir_list=args.train_noise_dir_list, target_dir_list=args.train_target_dir_list, volume_shape=args.volume_shape,
                                  batch_size=args.batch_size, slices=args.slices, image_size=args.image_size, degrees_max=args.degrees_max,
                                  distance_max=args.distance_max, nb_augmentations=args.train_nb_augmentations, crop_xy=args.crop_xy, crop_z=args.crop_z,
                                  offset_z=args.offset_z, padding=args.padding, noise_norm_constant=args.noise_norm_constant,
                                  target_norm_constant=args.target_norm_constant, nb_to_remove=args.nb_to_remove)
    
    v_gen = valDataGenerator(noise_dir_list=args.val_noise_dir_list, target_dir_list=args.val_target_dir_list, volume_shape=args.volume_shape,
                             nb_val=args.nb_val, slices=args.slices, degrees_max=args.degrees_max, distance_max=args.distance_max,
                             nb_augmentations=args.val_nb_augmentations, crop_xy=args.crop_xy, crop_z=args.crop_z, offset_z=args.offset_z,
                             padding=args.padding, noise_norm_constant=args.noise_norm_constant, target_norm_constant=args.target_norm_constant,
                             nb_to_remove=args.nb_to_remove)

    if args.disp_examples == 'True':
        print('Displaying examples')
        nb_examples = 3
        PlotGenerators(t_gen, v_gen, nb_examples)

    output_path.mkdir(parents=True, exist_ok=True)
    callbacks.append(LearningRateScheduler(schedule=Schedule(args.nb_epochs, args.lr)))
    callbacks.append(ModelCheckpoint(str(args.output_path) + "/weights.{epoch:03d}.hdf5",
                                     monitor="val_PSNR",
                                     verbose=1,
                                     mode="max",
    #                                 save_best_only=True))
                                     save_freq=int(args.save_freq*args.steps)))

    #Start training
    hist = model.fit(t_gen,
                    steps_per_epoch=args.steps,
                    epochs=args.nb_epochs,
                    validation_data=v_gen,
                    verbose=1,
                    callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)
                        


if __name__ == '__main__':
    main()    

