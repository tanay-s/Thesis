import tensorflow as tf
from utils import *
from ResNet import ResNet
import os
from config import args
import h5py
from validation import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

def test():
    # load the test data
    X_test, Y_test = load_data(args.img_w, args.n_cls, args.n_ch, mode='test')
    print('Test set', X_test.shape, Y_test.shape)
    # load the model
    model = ResNet()
    model.inference().accuracy_func().loss_func().train_func().pred_func()

    saver = tf.train.Saver()
    save_path = os.path.join(args.load_dir, '20180403-190256/' + 'model_39')
    w_plus = (Y_test.shape[0] - np.sum(Y_test, axis=0)) / (np.sum(Y_test, axis=0))

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        print("Model restored.")
        acc_valid, auroc_valid, precision_valid, recall_valid, f1_score_valid, loss_valid = validation(X_test, Y_test, args.val_batch_size, args.n_cls, sess, model, 0, 0, w_plus)

        print("AUROC: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, F1: {3:.4f}".format(np.mean(auroc_valid), np.mean(precision_valid), np.mean(recall_valid), np.mean(f1_score_valid)))

if __name__ == '__main__':
    test()