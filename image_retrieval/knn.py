import h5py
from test import *
from utils import *
from config import args
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import precision_score
import cv2

def get_db_features():
    h5f_train = h5py.File('/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/ResNet-50/cbir/DB_E25_features.h5', 'r')
    x_train = h5f_train['X_train'][:]
    y_train = h5f_train['Y_train'][:]
    features = h5f_train['features'][:]
    h5f_train.close()
    return x_train, y_train, features

def get_img_features(check_label=None):
    # x_test, y_test = load_data(args.img_w, args.n_cls, args.n_ch, mode='test')
    h5f_train = h5py.File('/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/retrieved_images/realimg_cond_'+str(check_label)+'.h5', 'r')
    #h5f_train = h5py.File(
     #    '/data/hula/tanay/CXR8/chest256_train_801010_no_normal.h5', 'r')
    x_test = h5f_train['X_test'][:]
    y_test = h5f_train['Y_test'][:]
    # features = h5f_train['features'][:]
    h5f_train.close()
    # y_test = np.zeros((x_test.shape[0], 14))
    # y_test[:, 5] = 1
    img = x_test#/255.   #make sure x is scaled between 0 and 1
    label = y_test
    features = test(img, label, single_img=0, all=0)
    return x_test, label, features #, y_t

def knn(x_train, y_train, db_features, img, label, test_features, check_label, k=5):
    img = np.reshape(img,(-1, args.img_w, args.img_h, 1))
    label = np.reshape(label, (-1, args.n_cls))
    test_features = np.reshape(test_features, (-1, 2048))
    database = zip(x_train, y_train, db_features)
    test_imgs = zip(img, label, test_features)
    map = []
    for test_iterator, test_tuple in enumerate(test_imgs):
        test_img = test_tuple[0]*255.
        cv2.imwrite('/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/retrieved_images/E25/'+str(check_label)+'/test_img_' + str(test_iterator) + '.png', test_img)
        distances = []

        for db_iterator, db_tuple in enumerate(database):
            distance = mean_squared_error(test_tuple[2], db_tuple[2])
            distances.append((distance, db_iterator))

        distances.sort()
        nn = distances[:k]
        nn_label = np.zeros((k, args.n_cls))
        for iterator, tuple in enumerate(nn):
            img = database[tuple[1]][0]#*255.
            nn_label[iterator, :] = np.reshape(database[tuple[1]][1], (-1, args.n_cls))
            cv2.imwrite('/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/retrieved_images/E25/'+str(check_label)+'/test_img_' + str(test_iterator) + '_nn_' + str(iterator) + '.png', img)
        count = 0.0
        avg_precision = []
        for i in range(nn_label.shape[0]):
            # index = np.where(test_tuple[1] == 1)
            if nn_label[i, check_label] == 1:
                count += 1
                avg_precision.append(count / (i+1))
            # else:
            #     avg_precision.append(count / (i+1))
        if sum(avg_precision) > 0:
            map.append(sum(avg_precision) / float(len(avg_precision)))
        else:
            map.append(0)
        # labels = np.concatenate((np.reshape(label[test_iterator, :], (-1, args.n_cls)), nn_label), axis=0)
    return sum(map)/len(map)

        # np.savetxt('/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/retrieved_images/labels_'+str(test_iterator)+'.csv', labels, delimiter=',')

def knn_precision(x_train, y_train, db_features, img, label, test_features, k=1):
    img = np.reshape(img, (-1, args.img_w, args.img_h, 1))
    label = np.reshape(label, (-1, args.n_cls))
    test_features = np.reshape(test_features, (-1, 2048))
    database = zip(x_train, y_train, db_features)
    test = zip(img, label, test_features)
    nn_label = np.zeros((img.shape[0], args.n_cls))

    for test_iterator, test_tuple in enumerate(test):
        distances = []
        for db_iterator, db_tuple in enumerate(database):
            distance = mean_squared_error(test_tuple[2], db_tuple[2])
            distances.append((distance, db_iterator))

        distances.sort()
        nn = distances[:k]
        for iterator, tuple in enumerate(nn):
            nn_label[test_iterator, :] = np.reshape(database[tuple[1]][1], (-1, args.n_cls))

    precision = []
    for i in range(args.n_cls):
        precision.append(precision_score(label[:, i], nn_label[:, i]))

    print('Precision:', precision)

def check_GAN(x_train, y_train, db_features, x_gan, y_gan, gan_features):
    x_gan = np.reshape(x_gan, (-1, args.img_w, args.img_h, 1))
    y_gan = np.reshape(y_gan, (-1, args.n_cls))
    gan_features = np.reshape(gan_features, (-1, 2048))
    database = zip(x_train, y_train, db_features)
    gan_imgs = zip(x_gan, y_gan, gan_features)
    mse = 0
    count = 0
    acc = np.zeros((64, 14))
    for gan_it, gan_tuple in enumerate(gan_imgs):
        distances = []
        for db_iterator, db_tuple in enumerate(database):
            distance = mean_squared_error(gan_tuple[2], db_tuple[2])
            distances.append((distance, db_iterator))

        distances.sort()
        nn = distances[0]
        nn_label = database[nn[1]][1]
        gan_label = gan_tuple[1]
        # index = np.where(gan_label == 1)
        # if gan_label[index] == nn_label[index]:
        #     count += 1
        tmp = (gan_label == nn_label)
        truth = np.array([int(x) for x in tmp])
        acc[gan_it, :] = truth


        mse += mean_squared_error(nn_label, gan_label)
    # accuracy = count / float(x_gan.shape[0])
    accuracy = np.mean(acc, 0)
    return mse, accuracy



if __name__ == '__main__':
    x_db, y_db, features_db = get_db_features()
    # x, y, features = get_img_features()
     #h5f = h5py.File('/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/ResNet-50/cbir/DB_mix_GAN_fm_E25_features.h5', 'w')
     #h5f.create_dataset('X_train', data=x)
     #h5f.create_dataset('Y_train', data=y)
     #h5f.create_dataset('features', data=features)
     #h5f.close()
    check_label = 12
    X_test, Y_test, features_test = get_img_features(check_label)
    # mse, accuracy = check_GAN(x_db, y_db, features_db, x_gan, y_gan, features_gan)
    # print mse
    # print accuracy
    map = knn(x_db, y_db, features_db, X_test, Y_test, features_test, check_label, k=5)
    print(map, check_label)














