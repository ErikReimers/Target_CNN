import numpy as np
import os
from os.path import exists
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from scipy import ndimage
import time
import random
from tensorflow.keras.utils import Sequence
import itertools
from matplotlib.widgets import RangeSlider



##########################################   Generator for the training data   ####################################################
class trainingDataGenerator(Sequence):
        def __init__(self, noise_dir_list, target_dir_list, volume_shape=(128,128,89), batch_size=8,
                     slices=3, image_size=128, degrees_max=15, distance_max=4, nb_augmentations=1, crop_xy=128, crop_z=89,
                     offset_z=0, padding='zero', noise_norm_constant=1, target_norm_constant=1, nb_to_remove=0):

            self.crop_z = crop_z
            self.batch_size = batch_size
            self.image_size = image_size
            self.slices = slices
            
            print('')
            print('')
            print('---------- INITIALIZING TRAINING DATA ----------')
            
            #Load in the noise volumes and crop them
            noise_volumes, _ = readDirectory(noise_dir_list, volume_shape, noise_norm_constant)
            noise_volumes = removeEarlyData(noise_volumes, nb_to_remove)
            
            #Save pre-padded dimensions 
            _, initial_nb_volumes, _, _, self.d = noise_volumes.shape

            #Pad the data spatially
            noise_volumes = padData(noise_volumes, slices, padding)
            
            #Make the target data
            #target_volumes = makeTargetVolumes(noise_volumes)

            #Load in the target volumes and crop them
            target_volumes, _ = readDirectory(target_dir_list, volume_shape, target_norm_constant)

            #Pad the target data spatially
            target_volumes = padData(target_volumes, slices, padding)

            #Duplicate the target data to have a 1:1 copy with each noise volume
            target_volumes = np.repeat(target_volumes, initial_nb_volumes, axis=1)
            
            #Augment the noise and target data
            noise_volumes, target_volumes = augmentData(noise_volumes, target_volumes, nb_augmentations, degrees_max, distance_max)

            #Crop to remove bleed over
            noise_volumes = cropData(noise_volumes, crop_xy, crop_z, offset_z)
            target_volumes = cropData(target_volumes, crop_xy, crop_z, offset_z)

            #Add back in zero padding
            self.noise_volumes = padData(noise_volumes, slices, 'zero')
            self.target_volumes = padData(target_volumes, slices, 'zero')

            #Save post-agumentation and post-crop dimensions 
            self.nb_dir, self.nb_volumes, self.w, self.h, _ = noise_volumes.shape

        
        #len shows how many unique batches you can make from the data
        def __len__(self):
                
            length = self.crop_z*self.nb_volumes*self.nb_dir//self.batch_size
            print('length of training generator: ', length)
            return length


        #getitem returns a batch worth of image patch pairs
        def __getitem__(self, idx):
                
            #x will be the noise patches and y will be the cooresponding target patches
            x = np.zeros((self.batch_size, self.image_size, self.image_size, self.slices), dtype=np.float32)
            y = np.zeros((self.batch_size, self.image_size, self.image_size, self.slices), dtype=np.float32)

            #Keep track of the number of saved patch pairs
            patch_counter = 0

            #Keep creating patch pairs until the counter equals the batch size
            while True:
                    
                #Pick a random directory number
                rand_dir_nb = random.randrange(self.nb_dir)
                
                #Pick a random volume number
                rand_vol_nb = random.randrange(self.nb_volumes)
                
                #Pick a random slice number (adjust for padding)
                rand_axial_slice_nb = random.randrange(self.d) + self.slices//2

                #Select the corresponding noise and target volumes
                noise_volume = self.noise_volumes[rand_dir_nb, rand_vol_nb, :, :, :]
                target_volume = self.target_volumes[rand_dir_nb, rand_vol_nb, :, :, :]
                
                if np.any(noise_volume):
                    #Grab the corresponding slice (2.5D)
                    slice_x = grabSlice(noise_volume, self.slices, rand_axial_slice_nb)
                    slice_y = grabSlice(target_volume, self.slices, rand_axial_slice_nb)

                    #Further augmentation by flipping left and right
                    rand_flip = random.randrange(2)
                    if rand_flip == 1:
                        slice_x = np.flip(slice_x,axis=1)
                        slice_y = np.flip(slice_y,axis=1)

                    #Randomly select the image size within (x and y dimensions)
                    i = np.random.randint(self.h - self.image_size + 1)
                    j = np.random.randint(self.w - self.image_size + 1)

                    #Save the patch pairs
                    x[patch_counter,:,:,:] = slice_x[i:i + self.image_size, j:j + self.image_size, :]
                    y[patch_counter,:,:,:] = slice_y[i:i + self.image_size, j:j + self.image_size, :]
                                
                    #Increment the counter                
                    patch_counter += 1
               
                    #Once the counter reaches the batch_size return the set of noisy pairs
                    if patch_counter == self.batch_size:
                        return x,y



##########################################   Generator for the validation data   ####################################################
class valDataGenerator(Sequence):
        def __init__(self, noise_dir_list, target_dir_list, volume_shape=(128,128,89), nb_val=8,
                     slices=3, degrees_max=15, distance_max=4, nb_augmentations=1, crop_xy=128, crop_z=89,
                     offset_z=0, padding='zero', noise_norm_constant=1, target_norm_constant=1, nb_to_remove=0):
                
            self.data = []
            self.nb_val = nb_val

            print('')
            print('')
            print('---------- INITIALIZING VALIDATION DATA ----------')
            
            #Load in the noise volumes and crop them
            noise_volumes, _ = readDirectory(noise_dir_list, volume_shape, noise_norm_constant)
            noise_volumes = removeEarlyData(noise_volumes, nb_to_remove)

            #Save pre-padded dimensions 
            _, initial_nb_volumes, _, _, d = noise_volumes.shape

            #Pad the data spatially
            noise_volumes = padData(noise_volumes, slices, padding)
            
            #Make the target data
            #target_volumes = makeTargetVolumes(noise_volumes)

            #Load in the target volumes and crop them
            target_volumes, _ = readDirectory(target_dir_list, volume_shape, target_norm_constant)

            #Pad the target data spatially
            target_volumes = padData(target_volumes, slices, padding)

            #Duplicate the target data to have a 1:1 copy with each noise volume
            target_volumes = np.repeat(target_volumes, initial_nb_volumes, axis=1)
            
            #Augment the noise and target data
            noise_volumes, target_volumes = augmentData(noise_volumes, target_volumes, nb_augmentations, degrees_max, distance_max)

            #Crop to remove bleed over
            noise_volumes = cropData(noise_volumes, crop_xy, crop_z, offset_z)
            target_volumes = cropData(target_volumes, crop_xy, crop_z, offset_z)

            #Add back in zero padding
            noise_volumes = padData(noise_volumes, slices, 'zero')
            target_volumes = padData(target_volumes, slices, 'zero')

            #Save the number of volumes post-agumentation
            nb_dir, nb_volumes, w, h, _ = noise_volumes.shape

            #Save the validation patch pairs
            for ii in range(nb_val):
                noise_volume = np.zeros((w,h,d),dtype=np.float32)
                while np.any(noise_volume) == False:
                    #Pick a random directory number
                    rand_dir_nb = random.randrange(nb_dir)
                
                    #Pick a random volume number
                    rand_vol_nb = random.randrange(nb_volumes)
                
                    #Pick a random slice number (adjust for padding)
                    rand_axial_slice_nb = random.randrange(d) + slices//2

                    #Select the corresponding noise and target volumes
                    noise_volume = noise_volumes[rand_dir_nb, rand_vol_nb, :, :, :]
                    target_volume = target_volumes[rand_dir_nb, rand_vol_nb, :, :, :]

                
                    #Grab the corresponding slice (2.5D)
                    slice_x = np.expand_dims(grabSlice(noise_volume, slices, rand_axial_slice_nb), 0)
                    slice_y = np.expand_dims(grabSlice(target_volume, slices, rand_axial_slice_nb), 0)

                    #Save the patch pairs
                    self.data.append([slice_x, slice_y])

            #Delete the initial volumes to save RAM
            del noise_volumes
            del target_volumes
                
        #len shows how many validation patch pairs were saved
        def __len__(self):
            return self.nb_val

        #getitem returns the validation patch pairs
        def __getitem__(self, idx):
            return self.data[idx]


      
##########################################      Functions      ####################################################
#Grab a slice from a volume (2.5D)
def grabSlice(volume, slices, axial_slice_nb):
    #print(axial_slice_nb-slices//2, axial_slice_nb+1+slices//2)
    return volume[:, :, (axial_slice_nb - slices//2):(axial_slice_nb + 1 + slices//2)]

#After augmenting the data, there will be bleed over into the padded areas, zero this out
def zeroOutData(volumes, z_crop):
    nb_dir, nb_volumes, w, h, d = volumes.shape      

    #Run through each volume and zero them
    for ii in range(nb_dir):
        for jj in range(nb_volumes):
            print(f'zeroing out volume augmentation bleed over {ii+1}/{nb_dir}, {jj+1}/{nb_volumes}   ', end = '\r')
            volumes[ii, jj, :, :, :] = zeroOutVolume(volumes[ii, jj, :, :, :], z_crop)
    nb_dir, nb_volumes, w, h, d = volumes.shape
    print('')
    print(f'Zeroed out data shape: {volumes.shape}  ')
    print('')
    return volumes


def zeroOutVolume(volume, z_crop, offset_z):
    w,h,d = volume.shape
    volume[:,:,0:(slices//2)] = 0
    volume[:,:,(d-slices//2):] = 0
    return volume


#Augment the data n times to also include rotations and translations
def augmentData(noise_volumes, target_volumes, nb_augmentations, degrees_max, distance_max):
    nb_dir, nb_volumes, w, h, d = noise_volumes.shape
    print(f'Augmenting data {nb_augmentations} times to also include rotations and translations')
    
    #New data will have (n+1) times the number of volumes (includes the 1 unaugmented set)
    augmented_noise_volumes = np.zeros((nb_dir, nb_volumes * (nb_augmentations + 1), w, h, d), dtype=np.float32)
    augmented_target_volumes = np.zeros((nb_dir, nb_volumes * (nb_augmentations + 1), w, h, d), dtype=np.float32)

    #First set is the original, unaugmented data
    augmented_noise_volumes[:, 0:nb_volumes, :, :, :] = noise_volumes
    augmented_target_volumes[:, 0:nb_volumes, :, :, :] = target_volumes

    #Run through the data n times and augment it
    for ii in range(nb_augmentations):
        print(f'Performing Augmention: {ii+1}/{nb_augmentations}                 ', end = '\r')
        augmented_noise_volumes[:, (ii+1)*nb_volumes:(ii+2)*nb_volumes, :, :, :], augmented_target_volumes[:, (ii+1)*nb_volumes:(ii+2)*nb_volumes, :, :, :] = randomAugmentationData(noise_volumes, target_volumes, degrees_max, distance_max)

    print('')
    print(f'Final augmented training data shape: {augmented_noise_volumes.shape}')
    print('')
    
    return augmented_noise_volumes, augmented_target_volumes



#Augment two sets of volumes with each volume pair getting a random augmentation
def randomAugmentationData(volumes_1, volumes_2, degrees_max, distance_max):
    nb_dir, nb_volumes, w, h, d = volumes_1.shape

    #Run through each volume pair and randomly augment them
    for ii in range(nb_dir):
        for jj in range(nb_volumes):
            volumes_1[ii, jj, :, :, :], volumes_2[ii, jj, :, :, :] = randomAugmentationVolume(volumes_1[ii, jj, :, :, :], volumes_2[ii, jj, :, :, :], degrees_max, distance_max)

    return volumes_1, volumes_2



#Do a double augmentation with random degree and distance
def randomAugmentationVolume(volume_1, volume_2, degrees_max, distance_max):
        
    degrees = np.zeros((3), dtype=np.float32)
    distance = np.zeros((3), dtype=np.float32)
    
    #Randomly pick a 3D rotation and translation up to the specified max values
    for ii in range(3):
        degrees[ii] = random.uniform(-1, 1) * degrees_max
        distance[ii] = random.uniform(-1, 1) * distance_max

    #Perform the augmentation
    volume_1 = setAugmentationVolume(volume_1, degrees, distance)
    volume_2 = setAugmentationVolume(volume_2, degrees, distance)
    return volume_1, volume_2

#Do a set augmentation to a set of volumes
def setAugmentationData(volumes, degrees, distance):
    nb_dir, nb_volumes, w, h, d = volumes.shape

    #Run through each volume pair and augment them
    for ii in range(nb_dir):
        for jj in range(nb_volumes):
            volumes[ii, jj, :, :, :] = setAugmentationVolume(volumes[ii, jj, :, :, :], degrees, distance)

    return volumes
    

#Do a single augmentation with a set degree and distance
def setAugmentationVolume(volume, degrees, distance):
    volume = rotateVolume(volume, degrees)
    volume = translateVolume(volume, distance)
    return volume



#Rotate the volume in 3D. Somewhat slow, may consider switching to np.mapcoordinates or sitk.Euler3DTransform
def rotateVolume(volume,degrees):
    
    volume = ndimage.rotate(volume, degrees[0], axes=(0,1), order = 0, reshape=False)
    volume = ndimage.rotate(volume, degrees[1], axes=(1,2), order = 0, reshape=False)
    volume = ndimage.rotate(volume, degrees[2], axes=(2,0), order = 0, reshape=False)
    return volume



#translate the volume in 3D
def translateVolume(volume, distance):
    volume = ndimage.shift(volume, distance, order = 0)
    return volume



#Make the target volumes from the noise data (average all volumes in a set)
def makeTargetVolumes(volumes):
    print('Making target volumes as average of noise volumes')
    nb_dir, nb_volumes, w, h, d = volumes.shape
    target_volumes = np.expand_dims(np.mean(volumes, axis=1), 1)
    target_volumes = np.repeat(target_volumes, nb_volumes, axis=1)
    print(f'Target volumes data shape: {target_volumes.shape}  ')
    print(f'Range of targets, min: {np.min(target_volumes)}, max: {np.max(target_volumes)}')
    print('')
    return target_volumes



#Take a set of volumes and pad them in space to be ready for 2.5D training
def padData(volumes, slices, padding):
    nb_dir, nb_volumes, w, h, d = volumes.shape        
    padded_volumes = np.zeros((nb_dir, nb_volumes, w, h, d+2*(slices//2)), dtype=np.float32)

    #Run through each volume and pad them
    for ii in range(nb_dir):
        for jj in range(nb_volumes):
            print(f'Padding volumes spacially {ii+1}/{nb_dir}, {jj+1}/{nb_volumes}   ', end = '\r')
            padded_volumes[ii, jj, :, :, :] = padVolume(volumes[ii, jj, :, :, :], padding, slices)
    volumes = padded_volumes
    nb_dir, nb_volumes, w, h, d = volumes.shape
    print('')
    print(f'Spacially padded data shape: {volumes.shape}  ')
    print('')
    return volumes



#Add slices to top and bottom to account for multislice 2.5D
def padVolume(volume,padding,slices):
    short_nb_slices = 7
    w,h,d = volume.shape
    
    #Check which padding option to use
    if padding in {'zero', 'zeros'}: #Add zeros to pad
        low_padding_slices = np.zeros((w, h, slices//2), dtype=np.float32)
        high_padding_slices = np.zeros((w, h, slices//2), dtype=np.float32)
        
    elif padding == 'mirror': #Mirror the image to pad
        low_padding_slices = np.flip(volume[:,:,:slices//2-1], 2)
        high_padding_slices = np.flip(volume[:,:,-slices//2:], 2)

    elif padding == 'short_mirror': #Mirror the last 7 slices of pixels back and forth
        low_padding_repeat = np.concatenate((volume[:,:,0:short_nb_slices], np.flip(volume[:,:,0:short_nb_slices], 2)), axis=2)
        high_padding_repeat = np.concatenate((np.flip(volume[:,:,-(short_nb_slices+1):-1], 2), volume[:,:,-(short_nb_slices+1):-1]), axis=2)

        low_padding_repeat = np.tile(low_padding_repeat, (slices//2)//(short_nb_slices*2) + 1)
        high_padding_repeat = np.tile(high_padding_repeat, (slices//2)//(short_nb_slices*2) + 1)
        
        low_padding_slices = low_padding_repeat[:,:,-(slices//2+1):-1]
        high_padding_slices = high_padding_repeat[:,:,0:(slices//2)]

    elif padding == 'random': #Randomly repeat the last 7 slices, ignore slice 0 and 1 because of noisey high-value voxels
        low_padding_repeat = volume[:,:,2:(short_nb_slices+2)]
        high_padding_repeat = volume[:,:,-(short_nb_slices+1):-1]

        low_padding_slices = np.zeros((w, h, slices//2), dtype=np.float32)
        high_padding_slices = np.zeros((w, h, slices//2), dtype=np.float32)

        for ii in range(slices//2):
            rand_slice_nb_low = random.randrange(short_nb_slices)
            rand_slice_nb_high = random.randrange(short_nb_slices)
            low_padding_slices[:,:,ii] = low_padding_repeat[:,:,rand_slice_nb_low]
            high_padding_slices[:,:,ii] = high_padding_repeat[:,:,rand_slice_nb_high]

    elif padding == 'repeat': #Repeat the last slice of pixels to pad
        low_padding_slices = np.repeat(volume[:,:,[0]], slices//2, axis=2)
        high_padding_slices = np.repeat(volume[:,:,[-1]], slices//2, axis=2)
        
    elif padding == 'average_repeat': #Repeat an average of the last 7 slices of pixels to pad
        low_padding_slices = np.repeat(np.expand_dims(np.mean(volume[:,:,0:short_nb_slices], axis=2),-1), slices//2, axis=2)
        high_padding_slices = np.repeat(np.expand_dims(np.mean(volume[:,:,-(short_nb_slices+1):-1], axis=2),-1), slices//2, axis=2)
            
    else:
        raise ValueError('use zero, mirror, short_mirror, random, or repeat, or average_repeat -ER')

    volume = np.concatenate((low_padding_slices, volume, high_padding_slices),axis=2)
    return volume



#Take a set of volumes and undo the cropping that was performed to them, returning them to original dimensions
def undoCropData(volumes, crop_xy, crop_z, offset_z, volume_shape):
    nb_dir, nb_volumes, w, h, d = volumes.shape
    uncropped_volumes = np.zeros((nb_dir, nb_volumes, volume_shape[0], volume_shape[1], volume_shape[2]), dtype=np.float32)

    #Run through each volume and undo the cropping
    for ii in range(nb_dir): 
        for jj in range(nb_volumes):
            print(f'Uncropping volumes {ii+1}/{nb_dir}, {jj+1}/{nb_volumes}  ', end = '\r')
            uncropped_volumes[ii, jj, :, :, :] = undoCropVolume(volumes[ii, jj, :, :, :],crop_xy, crop_z, offset_z, volume_shape)
    volumes = uncropped_volumes
    nb_dir, nb_volumes, w, h, d = volumes.shape
    print('')
    print(f'Uncropped data shape: {volumes.shape}')
    print('')
    return volumes



#Undo the cropping by padding with zeros
def undoCropVolume(volume, crop_xy, crop_z, offset_z, volume_shape):
    undo_crop_volume = np.zeros((volume_shape[0], volume_shape[1], volume_shape[2]), dtype=np.float32)
    w = volume_shape[0]
    h = volume_shape[1]
    d = volume_shape[2]
    xy_low = w//2 - crop_xy//2
    xy_high = w//2 + (crop_xy+1)//2
    z_low = d//2 - crop_z//2 + offset_z
    z_high = d//2 + (crop_z+1)//2 + offset_z
    if z_high > volume_shape[2] or z_low < 0:
        raise ValueError('offset_z value is outside possible range -ER')
    #print(f'Undo cropping: [{xy_low}:{xy_high}, {xy_low}:{xy_high}, {z_low}:{z_high}]')#, end = '\r')

    undo_crop_volume[xy_low:xy_high, xy_low:xy_high, z_low:z_high] = volume
    return undo_crop_volume



#Take a set of volumes and crop them in space to be ready for training
def cropData(volumes, crop_xy, crop_z, offset_z):
    nb_dir, nb_volumes, w, h, d = volumes.shape
    cropped_volumes = np.zeros((nb_dir, nb_volumes, crop_xy, crop_xy, crop_z), dtype=np.float32)
    for ii in range(nb_dir): #Run through the volumes and crop them
        for jj in range(nb_volumes):
            print(f'Cropping volumes {ii+1}/{nb_dir}, {jj+1}/{nb_volumes}  ', end = '\r')
            cropped_volumes[ii, jj, :, :, :] = cropVolume(volumes[ii, jj, :, :, :], crop_xy, crop_z, offset_z)
    volumes = cropped_volumes
    nb_dir, nb_volumes, w, h, d = volumes.shape
    print('')
    print(f'Cropped data shape: {volumes.shape}')
    print('')
    return volumes



#Crop the volume [x,y,z] into a smaller volume
def cropVolume(volume, crop_xy, crop_z, offset_z):
    w,h,d = volume.shape
    xy_low = w//2 - crop_xy//2
    xy_high = w//2 + (crop_xy+1)//2
    z_low = d//2 - crop_z//2 + offset_z
    z_high = d//2 + (crop_z+1)//2 + offset_z
    if z_high > d or z_low < 0:
        raise ValueError('offset_z value is outside possible range -ER')
    #print(f'Cropped: [{xy_low}:{xy_high}, {xy_low}:{xy_high}, {z_low}:{z_high}]')#, end = '\r')

    volume = volume[xy_low:xy_high, xy_low:xy_high, z_low:z_high]
    return volume

def removeEarlyNames(names, nb_to_remove):
    print(f'Removing first {nb_to_remove} names from the list of names')
    for ii in range(len(names)):
            names[ii] = names[ii][nb_to_remove:]
    #names = names[:][nb_to_remove:]
    print('')
    return names

#Remove the first few snaps from all the data
def removeEarlyData(volumes, nb_to_remove):
    print(f'Removing first {nb_to_remove} snaps from dataset')
    volumes = volumes[:, nb_to_remove:, :, :, :]
    print(f'Removed data shape: {volumes.shape}')
    print('')
    return volumes



#Save a set of volumes as .i files
def saveData(volumes, output_dir, filenames, suffix):
    nb_dir, nb_volumes, w, h, d = volumes.shape

    #Run through each volume and undo the cropping
    for ii in range(nb_dir): 
        for jj in range(nb_volumes):
            saveVolume(volumes[ii, jj, :, :, :], output_dir, filenames[ii][jj], suffix)



#Save a single volume as a .i file
def saveVolume(volume, output_dir, filename, suffix):

    #Make directory if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #Swap axes to z,y,x format
    volume = np.swapaxes(volume, 0, 2)

    #Set any negative values to zero
    volume[volume < 0] = 0

    #remove .i if it is there
    if filename[-2:] == '.i':
        filename = filename[:-2]

    #Add suffix    
    filename = filename + suffix + '.i'

    #Create the full path of the file
    full_path = os.path.join(output_dir, filename)

    #Save the volume
    volume.tofile(full_path)

    print('Saved: ',full_path)
    print('')

    

#Read either a single .i volume or a directory of volumes
#pass in a list of single .i volumes or directories
#Returns as [folder #, volume #, x, y, z]
def readDirectory(volume_dir_list, volume_shape, norm_constant):

    #Check if a .i file was specified or if a full directory was specified
    if volume_dir_list[0][-1] == 'i': 
        nb_dir = len(volume_dir_list)
        nb_volumes = 1
    else:
        nb_dir = len(volume_dir_list)
        nb_volumes = len(glob.glob(os.path.join(volume_dir_list[0], '*.i')))

    #Output the volumes and the coorespdoning individual volume names
    volumes = np.zeros((nb_dir , nb_volumes, volume_shape[0], volume_shape[1], volume_shape[2]), dtype=np.float32)
    volume_names = [[] for i in range(0,nb_dir)]
    
    #Run through each listed directed
    for ii in range(len(volume_dir_list)): 
        volume_dir = volume_dir_list[ii]
        print(f'Reading directory: {volume_dir}')

        #Check if directory or file exists
        if not exists(volume_dir): 
            raise ValueError(f'Directory or file {volume_dir} does not exist. -ER')

        #Check if a .i file was specified or if a full directory was specified
        if volume_dir[-1] == 'i':  
            #Read the single volume
            print(f'Reading volume: 1/1')
            volumes[ii,0,:,:,:] = readVolume(volume_dir, volume_shape)
            
            #Save the name of the volume
            volume_names[ii].append(os.path.basename(volume_dir))
            
        else:
            #Make a list of all the .i volumes in the directory    
            volume_paths = glob.glob(os.path.join(volume_dir, '*.i'))
            #Sort them by name to ensure they are read in as expected
            volume_paths = natsorted(volume_paths) 

            #Run through each volume within the directory
            for jj in range(len(volume_paths)):
                #Read the single volume
                print(f'Reading volume: {jj+1}/{len(volume_paths)}', end = '\r')
                volumes[ii,jj,:,:,:,] = readVolume(volume_paths[jj], volume_shape)

                #Save the name of the volume
                volume_names[ii].append(os.path.basename(volume_paths[jj]))
                
            print('')
    print(f'Read in data into shape: {volumes.shape}')
    print('')

    #Normalize the data
    volumes = normalizeData(volumes, norm_constant)
    
    print('')
    
    return volumes, volume_names

#Normalize the data by either the value value, or by a specified constant
def normalizeData(volumes, norm_constant):
    print(f'Range of raw data, min: {np.min(volumes)}, max: {np.max(volumes)}')
    print(f'Normalizing with constant (divide data by): {norm_constant}')
    
    volumes = volumes/norm_constant
    volumes[volumes > 1] = 1
    
    print(f'Range of Normalized data, min: {np.min(volumes)}, max: {np.max(volumes)}')
    print('')
    return volumes


def undoNormalizeData(volumes, norm_constant):
    print(f'Range of normalized data, min: {np.min(volumes)}, max: {np.max(volumes)}')
    print(f'Undoing normalization with constant (multiply data by): {norm_constant}') 

    volumes = volumes*norm_constant

    print(f'Range of unnormalized data, min: {np.min(volumes)}, max: {np.max(volumes)}')
    print('')
    return volumes

#Read a single .i volume
def readVolume(volume_path, volume_shape):
    fid = open(volume_path, 'r')
    data = np.fromfile(fid, dtype=np.float32)
    #rotate the data so that it follows the x y z dimensions instead of python's default z y x
    #2.5D Tensorflow requires that the third dimension of the image be the "colour channel"
    volume = np.reshape(data, [volume_shape[2], volume_shape[1], volume_shape[0]])
    return  np.swapaxes(volume, 0, 2)



#Quickly display a single volume
def quickDisplay(volume):
    w,h,d = volume.shape
    volume = np.swapaxes(volume, 0, 2)
    volume = np.flip(volume, axis = 0)
    
    plt.imshow(volume[:,h//2,:])
    plt.show()


    
#Quicky display a single volume as a MIP
def quickMIP(volume, MIP_axis):
    w,h,d = volume.shape
    volume = np.swapaxes(volume, 0, 2)
    volume = np.flip(volume, axis = 0)
    volume = np.amax(volume, axis = MIP_axis)
    
    plt.imshow(volume)
    plt.show()



#Quickly display an average of all the volumes in a stack
def quickAvDisplay(volumes):
    volume = np.squeeze(np.mean(volumes, axis = 0))
    w,h,d = volume.shape
    volume = np.swapaxes(volume, 0, 2)
    volume = np.flip(volume, axis = 0)
    plt.imshow(volume[:,h//2,:])
    plt.show()



#Plot the created generator objects
def PlotGenerators(t_gen, v_gen, nb_examples):

    #Ask for a batch of patch pairs from the training generator
    tmp_1, tmp_2 = t_gen[0]

    #Just hold onto the number of examples desired
    t_stack_1 = tmp_1[0:nb_examples, :, :, :]
    t_stack_2 = tmp_2[0:nb_examples, :, :, :]

    #convert to z,y,x format and put upright
    t_stack_1 = np.swapaxes(t_stack_1,1,3)
    t_stack_1 = np.flip(t_stack_1, axis = 1)
    t_stack_2 = np.swapaxes(t_stack_2,1,3)
    t_stack_2 = np.flip(t_stack_2, axis = 1)

    for ii in range(nb_examples):
        #Ask for a single patch pair from the validation generator
        v_stack_1, v_stack_2 = v_gen[ii]
        
        #convert to z,y,x format
        v_stack_1 = np.swapaxes(v_stack_1,1,3)
        v_stack_1 = np.flip(v_stack_1, axis = 1)
        v_stack_2 = np.swapaxes(v_stack_2,1,3)
        v_stack_2 = np.flip(v_stack_2, axis = 1)

        #Plot the patch pairs from the two generators
        plotExamplePairs(t_stack_1[ii,:,:,:], t_stack_2[ii,:,:,:], v_stack_1[0,:,:,:], v_stack_2[0,:,:,:])
    
#Plot the created generator objects
def plotExamplePairs(t_stack_1, t_stack_2, v_stack_1, v_stack_2):
    t_d, t_h, t_w = t_stack_1.shape
    v_d, v_h, v_w = v_stack_1.shape
    tracker = []
    fig, ax = plt.subplots(2, 4)
    plt.subplots_adjust(bottom=0.2)

    #The inital color bar setting when it loads
    init_percent = 0.2

    #Get the max value seen in all the patch pairs
    max_val = 0
    if np.max(t_stack_1) > max_val:
        max_val = np.max(t_stack_1)
    if np.max(t_stack_2) > max_val:
        max_val = np.max(t_stack_2)
    if np.max(v_stack_1) > max_val:
        max_val = np.max(v_stack_1)
    if np.max(v_stack_2) > max_val:
        max_val = np.max(v_stack_2)
        
    #plot 3D the patch pairs as two 2D views
    tracker.append(ax[0,0].imshow(t_stack_1[t_d//2,:,:],vmin=0,vmax=max_val*init_percent))
    tracker.append(ax[0,1].imshow(t_stack_2[t_d//2,:,:],vmin=0,vmax=max_val*init_percent))
    tracker.append(ax[0,2].imshow(t_stack_1[:,t_h//2,:],vmin=0,vmax=max_val*init_percent))
    tracker.append(ax[0,3].imshow(t_stack_2[:,t_h//2,:],vmin=0,vmax=max_val*init_percent))
    tracker.append(ax[1,0].imshow(v_stack_1[v_d//2,:,:],vmin=0,vmax=max_val*init_percent))
    tracker.append(ax[1,1].imshow(v_stack_2[v_d//2,:,:],vmin=0,vmax=max_val*init_percent))
    tracker.append(ax[1,2].imshow(v_stack_1[:,v_h//2,:],vmin=0,vmax=max_val*init_percent))
    tracker.append(ax[1,3].imshow(v_stack_2[:,v_h//2,:],vmin=0,vmax=max_val*init_percent))

    #Create the slider
    ax_slider = plt.axes([0.1, 0.01, 0.8, 0.05])
    slider = RangeSlider(ax_slider,'range',0,max_val,valinit=(0,max_val*init_percent))

    def update(val):
        for ii in range(8):
            tracker[ii].norm.vmin = val[0]
            tracker[ii].norm.vmax = val[1]
        fig.canvas.draw_idle()

    slider.on_changed(update)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == '__main__':

    #Example settings    
    noise_dir_list = ['/srv/Erik/Target_CNN/PETMR_images/DTBZ/S1/f1_as_1s']
    target_dir_list = ['/srv/Erik/Target_CNN/PETMR_images/DTBZ/S1/frames/f1.i']
    
    noise_norm_constant = 1
    target_norm_constant = 1
    volume_shape = (128,128,89)
    batch_size = 8
    nb_val = 8
    slices = 89
    image_size = 64
    degrees_max = 15
    distance_max = 4
    nb_training_augmentations = 0
    nb_validation_augmentations = 0
    crop_xy = 64
    crop_z = 0
    if crop_xy == 0:
        crop_xy = volume_shape[0]
    if crop_z == 0:
        crop_z = volume_shape[2]
    offset_z = 0
    padding = 'random'
    nb_examples = 7
    nb_to_remove = 0
##
#    noise_volumes, noise_names = readDirectory(noise_dir_list, volume_shape, norm_constant)
##    #print(noise_names)
##    noise_volumes = removeEarlyData(noise_volumes, nb_to_remove)
##    noise_volumes = cropData(noise_volumes, crop_xy, crop_z, offset_z)
#    noise_volumes = padData(noise_volumes, slices, padding)
#    snap = noise_volumes[0,59,:,:,:]
#    saveVolume(snap,'/srv/Erik/Target_CNN/Outputs/rots/','snap.i','')
#    snap_rot = setAugmentationVolume(snap, [0,0,10], [0,0,0])
#    snap_rot_zed = zeroOutVolume(snap_rot, slices)
#    saveVolume(snap_rot,'/srv/Erik/Target_CNN/Outputs/rots/','snap_rot_zed2.i','')
#    snap_rot_crop = cropVolume(snap_rot,128,89,0)
#    saveVolume(snap_rot_crop,'/srv/Erik/Target_CNN/Outputs/rots/','snap_rot_crop2.i','') 
##    target_volumes = makeTargetVolumes(noise_volumes)
##
##    noise_volumes, target_volumes = augmentData(noise_volumes, target_volumes, nb_training_augmentations, degrees_max, distance_max)
##
##    #quickAvDisplay(noise_volumes[0,:,:,:])
##    quickDisplay(noise_volumes[0,115,:,:,:])
##    quickDisplay(target_volumes[0,115,:,:,:])
##    quickDisplay(noise_volumes[0,50,:,:,:])
##    quickDisplay(target_volumes[0,50,:,:,:])
##    quickDisplay(noise_volumes[0,20,:,:,:])
##    quickDisplay(target_volumes[0,30,:,:,:])
##    quickDisplay(noise_volumes[0,40,:,:,:])
##    quickDisplay(target_volumes[0,40,:,:,:])
##    quickDisplay(noise_volumes[0,100,:,:,:])
##    quickDisplay(target_volumes[0,100,:,:,:])
##    quickDisplay(noise_volumes[0,170,:,:,:])
##    quickDisplay(target_volumes[0,170,:,:,:])
##    quickDisplay(noise_volumes[0,160,:,:,:])
##    quickDisplay(target_volumes[0,160,:,:,:])
##    
##    
##    volume = noise_volumes[0,30,:,:,:]
##    print(volume.shape)
##    #saveVolume(volume,'/srv/Erik/TestSave','test_volume.i')
##    saveData(noise_volumes[[0],60:120,:,:,:],'/srv/Erik/TestSave2',noise_names,'_test')
##
##    e_slice = grabSlice(volume, slices, 32+slices//2)
##    print(e_slice.shape)

    t_gen = trainingDataGenerator(noise_dir_list=noise_dir_list, target_dir_list=target_dir_list, volume_shape=volume_shape,
                                  batch_size=batch_size, slices=slices, image_size=image_size, degrees_max=degrees_max,
                                  distance_max=distance_max, nb_augmentations=nb_training_augmentations, crop_xy=crop_xy, crop_z=crop_z,
                                  offset_z=offset_z, padding=padding,noise_norm_constant=noise_norm_constant,target_norm_constant=target_norm_constant,
                                  nb_to_remove=nb_to_remove)

    
    v_gen = valDataGenerator(noise_dir_list=noise_dir_list, target_dir_list=target_dir_list, volume_shape=volume_shape,
                             nb_val=nb_val, slices=slices, degrees_max=degrees_max, distance_max=distance_max,
                             nb_augmentations=nb_validation_augmentations, crop_xy=crop_xy, crop_z=crop_z, #offset_z=offset_z,
                             padding=padding,noise_norm_constant=noise_norm_constant,target_norm_constant=target_norm_constant, nb_to_remove=nb_to_remove)


    PlotGenerators(t_gen, v_gen, nb_examples)  

