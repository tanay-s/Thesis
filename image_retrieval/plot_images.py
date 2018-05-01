import matplotlib.pyplot as plt
import h5py
import random
from knn import *
import glob
import cv2
#
h5f_train = h5py.File('/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/ResNet-50/features_no_normal_test.h5', 'r')
x_test = h5f_train['X_test'][:]
y_test = h5f_train['Y_test'][:]
features = h5f_train['features'][:]
h5f_train.close()

it = range(y_test.shape[0])
data = zip(it, y_test)
random.seed(110)
random.shuffle(data)


# for i in range(14):
#     for it, label in data:
#         if label[i] == 1:
#             if not it in img:
#                 img.append(it)
#                 break

for i in range(13):
    img = []
    count=0
    for it, label in data:
        if label[i] == 1:
            if not it in img:
                img.append(it)
                count+=1
        if count>=64:
            break
    testX = x_test[img, :]
    testY = y_test[img, :]
    testF = features[img, :]
    h5f = h5py.File('/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/retrieved_images/realimg_cond_'+str(i)+'.h5', 'w')
    h5f.create_dataset('X_test', data=testX)
    h5f.create_dataset('Y_test', data=testY)
    h5f.create_dataset('features', data=testF)
    h5f.close()
# conditions = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
#               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# fig = plt.figure()
# for i in range(14):
#     img = cv2.imread('/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/retrieved_images/test_img_'+str(i)+'.png')
#     img_plot = img[:, :, 0]
#     plt.subplot(6, 14, i+1)
#     plt.title(conditions[i])
#     plt.axis('off')
#     plt.imshow(img_plot, cmap='gray')
#
#
# for i in range(14):
#     files = glob.glob('/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/retrieved_images/test_img_'+str(i)+'_*')
#     files.sort()
#
#     for j, file in enumerate(files):
#         img = cv2.imread(file)
#         img_plot = img[:, :, 0]
#         plt.subplot(6, 14, (i+1)+(j+1)*14)
#         plt.title(conditions[i])
#         plt.axis('off')
#         plt.imshow(img_plot, cmap='gray')
# plt.show()


h5f_train = h5py.File('/data/hula/tanay/CXR8/chest256_train_801010_no_normal.h5', 'r')
# x_test = h5f_train['X_train'][:]
y_test = h5f_train['Y_train'][:]
h5f_train.close()

dir_path_train = '/data/hula/tanay/CXR8/gan_fm_data.h5'
h5f_test = h5py.File(dir_path_train, 'r')
# x_train2 = h5f_test['X_train'][:]
# x_train2 = (128 * x_train2 + 128) / 255
y_train2 = h5f_test['Y_train'][:]
h5f_test.close()

# labels = np.concatenate((y_test, y_train2))
images = np.sum(y_test, axis=0)

print(images/float(y_test.shape[0]))