"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     7/1/2017
Comments: Run this file to test the best saved model
**********************************************************************************
"""

import tensorflow as tf
from utils import *
from ResNet import ResNet
import os
from config import args
import h5py
from feature_extractor import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

def test(X_test=None, Y_test=None, single_img = 0, all=0):
    # load the model
    model = ResNet()
    model.inference().accuracy_func().loss_func().train_func().pred_func()

    saver = tf.train.Saver()
    save_path = '/data/hula/tanay/Codes/ChestX/image_retrieval/chest_xray/ResNet-50/checkpoints/20180404-211241/model_24'
    if single_img == 1:
        w_plus = np.zeros((14))
        w_plus[np.where(Y_test == 1)] = 1
        args.val_batch_size = 1
        print('Test set', X_test.shape, Y_test.shape)
    else:
        divisor = np.sum(Y_test, axis=0)
        nonzero_index = []
        zero_index = []
        for d in range(divisor.shape[0]):
            if divisor[d] != 0:
                nonzero_index.append(d)
            else:
                zero_index.append(d)
        dividend = Y_test.shape[0] - np.sum(Y_test, axis=0)
        for i in nonzero_index:
            dividend[i] = dividend[i]/divisor[i]
        for i in zero_index:
            dividend[i] = 1
        args.val_batch_size = X_test.shape[0]
        w_plus = dividend
        print('Test set', X_test.shape, Y_test.shape)

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        print("Model restored.")
        if all == 0:
            features = compute_features(X_test, Y_test, args.val_batch_size, sess, model, w_plus)
        else:
            features = np.zeros((X_test.shape[0], 2048))
            test_batch_size = 100
            batches = X_test.shape[0]/test_batch_size
            remain = X_test.shape[0]%test_batch_size
            for i in range(batches):
                img = X_test[i * test_batch_size:(i + 1) * test_batch_size, :]
                label = Y_test[i * test_batch_size:(i + 1) * test_batch_size, :]
                features[i * test_batch_size:(i + 1) * test_batch_size, :] = compute_features(img, label, test_batch_size, sess, model, w_plus)
            img = X_test[-remain:, :]
            label = Y_test[-remain:, :]
            features[-remain:, :] = compute_features(img, label, remain, sess, model, w_plus)

    return features

    # h5f = h5py.File('vol_features_no_normal_fix_weight.h5', 'w')
    # h5f.create_dataset('features', data=vol_features)
    # h5f.create_dataset('X_test', data=X_test[:features.shape[0]])
    # h5f.create_dataset('Y_test', data=Y_test[:features.shape[0]])
    # h5f.close()

    # h5f = h5py.File('cls_act_map_no_normal_fix_weight.h5', 'w')
    # h5f.create_dataset('cls_act_map', data=cls_act_map)
    # h5f.close()

if __name__ == '__main__':
    test()
