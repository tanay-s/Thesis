import cv2
import glob
import os
import numpy as np
import h5py

def separate_image(path):
    files = glob.glob(path+'/*.png')

    for file in files:
        word = file[73:]
        dot_index = word.find('.')
        name = word[:dot_index]
        directory = path + '/' + name
        if not os.path.exists(directory):
            os.makedirs(directory)
        big_img = cv2.imread(file)
        for j in range(8):
            for k in range(8):
                small_img = big_img[j*256:(j+1)*256, k*256:(k+1)*256]
                cv2.imwrite(directory+ '/' + str(k+j*8) + '.png', small_img)

def create_hdf(path):
    for i in range(14):
        images = np.zeros((64, 256, 256, 1))
        files = glob.glob(path+'/'+str(i)+'/*.png')
        for j, file in enumerate(files):
            img = cv2.imread(file)
            img = img[:, :, 0]
            img = np.reshape(img, (256, 256, 1))
            images[j, :] = img/255.
        labels = np.zeros((64, 14))
        labels[:, i] = 1
        h5f = h5py.File(path+'/'+str(i)+'/cond_'+str(i)+'.h5', 'w')
        h5f.create_dataset('X', data=images)
        h5f.create_dataset('Y', data=labels)
        h5f.close()

def single_img_label(path, big_img, dir_name): #for images with labels other than one hot(multi label)
    directory = path + '/' + dir_name
    if not os.path.exists(directory):
        os.makedirs(directory)

    for j in range(8):
        for k in range(8):
            small_img = big_img[j * 256:(j + 1) * 256, k * 256:(k + 1) * 256]
            cv2.imwrite(directory + '/' + str(k + j * 8) + '.png', small_img)

def single_img_label_hdf(path, name, label):
    images = np.zeros((64, 256, 256, 1))
    files = glob.glob(path + '/*.png')
    for j, file in enumerate(files):
        img = cv2.imread(file)
        img = img[:, :, 0]
        img = np.reshape(img, (256, 256, 1))
        images[j, :] = img / 255.
    labels = np.zeros((64, 14))
    labels[:] = label
    h5f = h5py.File(path + '/' + name + '.h5', 'w')
    h5f.create_dataset('X', data=images)
    h5f.create_dataset('Y', data=labels)
    h5f.close()

if __name__ == '__main__':
    path = '/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/GAN'
    # separate_image(path)
    label = np.array([1,0,1,1,0,0,0,0,0,0,0,0,1,0])

    img = cv2.imread('/data/hula/tanay/Codes/ChestX/Conditional_DCGAN/samples/test_arange_0.png')
    single_img_label(path, img, 'label')
    single_img_label_hdf(path, 'multicond', label)