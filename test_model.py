import argparse
import numpy as np
import numpy.matlib
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from model import get_model
import glob
import os
from PIL import Image
import tempfile
import imutils
import nibabel as nib
from scipy.ndimage import gaussian_filter
from natsort import natsorted
from scipy.io import savemat
from skimage.metrics import structural_similarity as ssim
from generator import readDirectory, removeEarlyData, removeEarlyNames, cropData, padData, setAugmentationData, grabSlice, undoCropData, undoNormalizeData, saveData, quickDisplay, zeroOutData
from matplotlib.widgets import RangeSlider



#Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description='Test trained model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_noise_dir_list', nargs='*', type=str, required=True,
                        help='test volume dir')
    parser.add_argument('--model', type=str, default='unet',
                        help='model architecture (srresnet or unet)')
    parser.add_argument('--weight_folder', type=str, required=True,
                        help='trained weight file')
    parser.add_argument('--output_dir', type=str, default='test_model_dump',
                        help='if set, save resulting images otherwise show result using imshow')
    parser.add_argument('--degrees', nargs='*', type=int, default=[0,0,0],
                        help='do you want to add rotations to the viewing?')
    parser.add_argument('--distance', nargs='*', type=int, default=[0,0,0],
                        help='do you want to move the viewing?')    
    parser.add_argument('--weights_nb',type=int, default=-1,
                        help='which weights file do you want to use, default (-1) will use the most recently saved')
    parser.add_argument('--volume_shape', nargs='*', type=int, default=[128,128,89],
                        help='What is the shape of volume?')
    parser.add_argument('--slices', type=int,default=3,
                        help='how many slices to include for 2.5D, odd numbers only')
    parser.add_argument('--padding',type=str,default='zero',
                        help='If you want to adding padding in the z direction to the volumes?')
    parser.add_argument('--crop_xy', type=int, default=0,
                        help='Would you like to crop the image to a size?, 0 will not crop')
    parser.add_argument('--crop_z', type=int, default=0,
                        help='Would you like to cropz the image to a size?, 0 will not crop_z')
    parser.add_argument('--offset_z', type=int, default=0,
                        help='Would you like to offset the image in the z direction?')
    parser.add_argument('--undo_crop',type=str,default='True',
                        help='Would you like to pad with zeros to undo the cropping?')
    parser.add_argument('--noise_norm_constant', type=float, default=1,
                        help='Normalization constant for the noisy data')
    parser.add_argument('--target_norm_constant', type=float, default=1,
                        help='Normalization constant to undo the normalization of the target')
    parser.add_argument('--nb_to_remove', type=int, default=0,
                        help='Number of snaps to remove from the beginning of the data')


    args = parser.parse_args()
    return args


def main():
    print('')
    print('')
    args = get_args()


    if args.crop_xy == 0:
        args.crop_xy = args.volume_shape[0]
    if args.crop_z == 0:
        args.crop_z = args.volume_shape[2]


    list_of_files = natsorted(glob.glob(args.weight_folder + '/*.hdf5'),key=str)
    #weight_file = max(list_of_files, key=os.path.getctime)
    weight_file = list_of_files[args.weights_nb]

    #Get the model from the model.py file
    model = get_model(model_name=args.model, slices=args.slices)
    print('weight file: ', weight_file)
    model.load_weights(weight_file)
    
    #if saving the images, make the specified folder
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)


    #Load in the noise volumes and crop them
    noise_volumes, noise_names = readDirectory(args.test_noise_dir_list, args.volume_shape, args.noise_norm_constant)
    noise_volumes = removeEarlyData(noise_volumes, args.nb_to_remove)
    noise_names = removeEarlyNames(noise_names, args.nb_to_remove)
    #noise_volumes = cropData(noise_volumes, args.crop_xy, args.crop_z, args.offset_z)

    #Save pre-padded dimensions 
    nb_dir, nb_volumes, w, h, d = noise_volumes.shape

    #Pad the data spatially
    noise_volumes = padData(noise_volumes, args.slices, args.padding)

    #Augment the noise data
    if args.degrees != [0,0,0] or args.distance != [0,0,0]:
        noise_volumes = setAugmentationData(noise_volumes, args.degrees, args.distance)

    #Crop to remove the bleed over
    noise_volumes = cropData(noise_volumes, args.crop_xy, args.crop_z, args.offset_z)

    #Add back in zero padding
    noise_volumes = padData(noise_volumes, args.slices, 'zero')

    #Save post-agumentation and post-crop dimensions 
    nb_dir, nb_volumes, w, h, _ = noise_volumes.shape

    input_volumes = np.zeros((nb_dir, nb_volumes, w, h, d), dtype=np.float32)
    denoised_volumes = np.zeros((nb_dir, nb_volumes, w, h, d), dtype=np.float32)

    #Run through all the data and "denoise" it
    for ii in range(nb_dir):
        for jj in range(nb_volumes):
            volume = noise_volumes[ii, jj, :, :, :]
            for kk in range(d):
                print(f'Denoising: {ii+1}/{nb_dir}, {jj+1}/{nb_volumes}, {kk+1}/{d}     ', end = '\r')

                #Grab the noise slice
                axial_slice_nb = kk + args.slices//2
                noise_slice = grabSlice(volume, args.slices, axial_slice_nb)

                #denoise the slice via the model
                denoised_slice = model.predict(np.expand_dims(noise_slice, 0))

                #quickDisplay(noise_slice)
                #quickDisplay(denoised_slice[0,:,:,:])

                #Save the slices into the volumes
                input_volumes[ii,jj,:,:,kk] = noise_slice[:,:,args.slices//2]
                denoised_volumes[ii,jj,:,:,kk] = denoised_slice[0,:,:,args.slices//2]
            print('')
                
    #Undo the cropping to return to original dimensions
    if args.undo_crop == 'True':
        input_volumes = undoCropData(input_volumes, args.crop_xy, args.crop_z, args.offset_z, args.volume_shape)
        denoised_volumes = undoCropData(denoised_volumes, args.crop_xy, args.crop_z, args.offset_z, args.volume_shape)
        
    #Undo the normalization
    print('Undoing normalization of input volumes')
    input_volumes = undoNormalizeData(input_volumes, args.noise_norm_constant)
    print('Undoing normalization of output volumes')
    denoised_volumes = undoNormalizeData(denoised_volumes, args.target_norm_constant)

    #Save the volumes
    saveData(input_volumes, args.output_dir, noise_names, '_noise')
    saveData(denoised_volumes, args.output_dir, noise_names, '_denoised')

            
if __name__ == '__main__':
    main()
