"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     7/1/2017
Comments: Run this file to train the resNet model and save the best trained model
**********************************************************************************
"""

import numpy as np
import tensorflow as tf
import time
from validation import validation
import csv
from ResNet import ResNet
from utils import *
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

conditions = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
              'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def train():
    # Loading the ChestXray data
    X_train, Y_train, X_valid, Y_valid = load_data(args.img_w, args.n_cls, args.n_ch, mode='train', with_normal=False)
    print('Training set', X_train.shape, Y_train.shape)
    print('Validation set', X_valid.shape, Y_valid.shape)

    # Creating the ResNet model
    model = ResNet()
    model.inference().accuracy_func().loss_func().train_func().pred_func()

    # Saving the best trained model (based on the validation accuracy)
    saver = tf.train.Saver()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, 'model_')
    best_validation_accuracy = 0

    # create csv files
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    create_acc_loss_file(conditions)
    create_precision_recall_file(conditions)

    w_plus = (Y_train.shape[0] - np.sum(Y_train, axis=0)) / (np.sum(Y_train, axis=0))

    loss_batch_all = np.array([])
    acc_batch_all = np.zeros((0, args.n_cls))
    logits_batch_all = np.zeros((0, args.n_cls))
    y_batch_all = np.zeros((0, args.n_cls))
    sum_count = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Initialized")
        merged = tf.summary.merge_all()
        batch_writer = tf.summary.FileWriter(args.logs_path + '/batch/', sess.graph)
        valid_writer = tf.summary.FileWriter(args.logs_path + '/valid/')
        for epoch in range(args.num_epoch):
            model.is_train = True
            epoch_start_time = time.time()
            print('__________________________________________________________________________'
                  '____________________________________________________________')
            print('--------------------------------------------------------Training, Epoch: {}'
                  ' -----------------------------------------------------------'.format(epoch + 1))
            print("Atlc\tCrdmg\tEffus\tInflt\tMass\tNodle\tPnum\tPntrx\tConsd"
                  "\tEdma\tEmpys\tFbrss\tTkng\tHrna\t|Avg.\t|Loss\t|Step")
            X_train, Y_train = randomize(X_train, Y_train)
            num_train_batch = int(Y_train.shape[0] / args.batch_size)
            for step in range(num_train_batch):
                start = step * args.batch_size
                end = (step + 1) * args.batch_size
                X_batch, Y_batch = get_next_batch(X_train, Y_train, start, end)
                # X_batch = random_rotation_2d(X_batch, 10.0)
                feed_dict_batch = {model.x: X_batch, model.y: Y_batch, model.w_plus: w_plus}

                _, acc_batch, logits_batch, loss_batch = sess.run([model.train_op,
                                                                     model.accuracy,
                                                                     model.get_logits,
                                                                   model.loss],
                                                            feed_dict=feed_dict_batch)

                acc_batch_all = np.concatenate((acc_batch_all, acc_batch.reshape([1, args.n_cls])))
                logits_batch_all = np.concatenate((logits_batch_all, logits_batch))
                y_batch_all = np.concatenate((y_batch_all, Y_batch))
                loss_batch_all = np.append(loss_batch_all, loss_batch)
                if (step % args.report_freq==0 and step!=0):
                    mean_auroc_cond = auroc_generator(y_batch_all, logits_batch_all)
                    precision, recall, f1_score = prf_generator(y_batch_all, logits_batch_all)
                    mean_acc_cond = np.mean(acc_batch_all, axis=0)
                    # mean_auroc_cond = np.mean(auroc_batch_all, axis=0)
                    mean_loss = np.mean(loss_batch_all)

                    for i in mean_auroc_cond:
                        print '{:.2f}\t'.format(i),
                    print '|{0:.2f}\t|{1:0.02}\t|{2}'.format(np.mean(mean_auroc_cond), mean_loss, step)
                    summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Mean_Accuracy',
                                                                    simple_value=np.mean(mean_acc_cond) * 100),
                                                   tf.Summary.Value(tag='Mean_AUROC',
                                                                    simple_value=np.mean(mean_auroc_cond)),
                                                   tf.Summary.Value(tag='Mean_Precision',
                                                                    simple_value=np.mean(precision)),
                                                   tf.Summary.Value(tag='Mean_Recall',
                                                                    simple_value=np.mean(recall)),
                                                   tf.Summary.Value(tag='Mean_F1_score',
                                                                    simple_value=np.mean(f1_score))
                                                   ])
                    batch_writer.add_summary(summary_tr, sum_count * args.report_freq)
                    for cond in range(args.n_cls):
                        with tf.name_scope('Accuracy'):
                            summary_tr = tf.Summary(
                                value=[tf.Summary.Value(tag='Accuracy_' + conditions[cond],
                                                        simple_value=mean_acc_cond[cond] * 100)])
                            batch_writer.add_summary(summary_tr, sum_count * args.report_freq)
                        with tf.name_scope('AUROC'):
                            summary_tr = tf.Summary(
                                value=[tf.Summary.Value(tag='AUROC_' + conditions[cond],
                                                        simple_value=mean_auroc_cond[cond])])
                            batch_writer.add_summary(summary_tr, sum_count * args.report_freq)
                        with tf.name_scope('Precision'):
                            summary_tr = tf.Summary(
                                value=[tf.Summary.Value(tag='Precision_' + conditions[cond],
                                                        simple_value=precision[cond])])
                            batch_writer.add_summary(summary_tr, sum_count * args.report_freq)
                        with tf.name_scope('Recall'):
                            summary_tr = tf.Summary(
                                value=[tf.Summary.Value(tag='Recall_' + conditions[cond],
                                                        simple_value=recall[cond])])
                            batch_writer.add_summary(summary_tr, sum_count * args.report_freq)
                        with tf.name_scope('F1_score'):
                            summary_tr = tf.Summary(
                                value=[tf.Summary.Value(tag='F1_score_' + conditions[cond],
                                                        simple_value=f1_score[cond])])
                            batch_writer.add_summary(summary_tr, sum_count * args.report_freq)


                    summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Mean_Loss', simple_value=mean_loss)])
                    batch_writer.add_summary(summary_tr, sum_count * args.report_freq)
                    summary = sess.run(merged, feed_dict=feed_dict_batch)
                    batch_writer.add_summary(summary, sum_count * args.report_freq)
                    sum_count += 1
                    loss_batch_all = np.array([])
                    acc_batch_all = np.zeros((0, args.n_cls))
                    logits_batch_all = np.zeros((0, args.n_cls))
                    y_batch_all = np.zeros((0, args.n_cls))

            acc_valid, auroc_valid, precision_valid, recall_valid, f1_score_valid, loss_valid = validation(X_valid, Y_valid, args.val_batch_size, args.n_cls,
                                                       sess, model, epoch, epoch_start_time, w_plus)

            for cond in range(args.n_cls):
                summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Accuracy_' + conditions[cond],
                                                                   simple_value=acc_valid[cond] * 100)])
                valid_writer.add_summary(summary_valid, sum_count * args.report_freq)
                summary_valid = tf.Summary(value=[tf.Summary.Value(tag='AUROC_' + conditions[cond],
                                                                   simple_value=auroc_valid[cond])])
                valid_writer.add_summary(summary_valid, sum_count * args.report_freq)
                summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Precision_' + conditions[cond],
                                                                   simple_value=precision_valid[cond])])
                valid_writer.add_summary(summary_valid, sum_count * args.report_freq)
                summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Recall_' + conditions[cond],
                                                                   simple_value=recall_valid[cond])])
                valid_writer.add_summary(summary_valid, sum_count * args.report_freq)
                summary_valid = tf.Summary(value=[tf.Summary.Value(tag='F1_score_' + conditions[cond],
                                                                   simple_value=f1_score_valid[cond])])
                valid_writer.add_summary(summary_valid, sum_count * args.report_freq)

            summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Mean_Loss', simple_value=loss_valid)])
            valid_writer.add_summary(summary_valid, sum_count * args.report_freq)

            summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Mean_Accuracy', simple_value=np.mean(acc_valid)*100)])
            valid_writer.add_summary(summary_valid, sum_count * args.report_freq)

            summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Mean_AUROC', simple_value=np.mean(auroc_valid))])
            valid_writer.add_summary(summary_valid, sum_count * args.report_freq)

            summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Mean_Precision', simple_value=np.mean(precision_valid))])
            valid_writer.add_summary(summary_valid, sum_count * args.report_freq)

            summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Mean_Recall', simple_value=np.mean(recall_valid))])
            valid_writer.add_summary(summary_valid, sum_count * args.report_freq)

            summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Mean_F1_score', simple_value=np.mean(f1_score_valid))])
            valid_writer.add_summary(summary_valid, sum_count * args.report_freq)

            # save the model after each epoch
            saver.save(sess=sess, save_path=save_path + str(epoch))

            if np.mean(auroc_valid) > best_validation_accuracy:
                # Update the best-known validation accuracy.
                saver.save(sess=sess, save_path=save_path + str(epoch)+ '_best_auroc')
                best_validation_accuracy = np.mean(auroc_valid)
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''
            epoch_time = time.time() - epoch_start_time
            print('---------------------------------Validation------------------------------------------')
            print("Epoch {0}, run time: {1:.1f} seconds, loss: {2:.2f}, auroc: {3:.2f}{4}"
                  .format(epoch + 1, epoch_time, loss_valid, np.mean(auroc_valid), improved_str))


if __name__ == '__main__':
    train()
