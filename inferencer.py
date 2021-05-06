"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""
import tensorflow as tf
from resolution_network import ResoNet
from solver import Solver
from easydict import EasyDict as edict
import cv2, yaml, os, dlib
from py_utils.vis import vis_im
import numpy as np
from py_utils.face_utils import lib
from py_utils.vid_utils import proc_vid as pv
import logging
import csv
import pandas as pd
import matplotlib.pyplot as plt

print('***********')
print('Detecting DeepFake images, prob == -1 denotes opt out')
print('***********')
# Parse config
cfg_file = 'cfgs/res50.yml'
with open(cfg_file, 'r') as f:
    cfg = edict(yaml.load(f))
sample_num = 10

# Employ dlib to extract face area and landmark points

front_face_detector = dlib.get_frontal_face_detector()
lmark_predictor = dlib.shape_predictor("/content/DeepFake_ArtifactWarper/dlib_model/shape_predictor_68_face_landmarks.dat")

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True
# init session
sess = tf.Session(config=tfconfig)
# Build network
reso_net = ResoNet(cfg=cfg, is_train=False)
reso_net.build()
# Build solver
solver = Solver(sess=sess, cfg=cfg, net=reso_net)
solver.init()


def im_test(im):
    face_info = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)
    # Samples
    if len(face_info) == 0:
        print("NOOO")
        logging.warning('No faces are detected.')
        prob = -1  # we ignore this case
    else:
        print("SIIII")
        # Check how many faces in an image
        logging.info('{} faces are detected.'.format(len(face_info)))
        max_prob = -1
        # If one face is fake, the image is fake
        for _, point in face_info:
            rois = []
            for i in range(sample_num):
                roi, _ = lib.cut_head([im], point, i)
                rois.append(cv2.resize(roi[0], tuple(cfg.IMG_SIZE[:2])))
            vis_im(rois, 'tmp/vis.jpg')
            prob = solver.test(rois)
            prob = np.mean(np.sort(prob[:, 0])[np.round(sample_num / 2).astype(int):])
            if prob >= max_prob:
                max_prob = prob
        prob = max_prob
    return prob


def run(input_dir):
    logging.basicConfig(filename='run.log', filemode='w', format='[%(asctime)s - %(levelname)s] %(message)s',
                        level=logging.INFO)
    
    prob_list = []
    i = 0
    for f_name in input_dir:
        print("{}/{}".format(i+1,len(input_dir)))
        i = i+1
        # Parse video
        f_path = os.path.join(f_name)
        #print('Testing: ' + f_path)
        logging.info('Testing: ' + f_path)
        suffix = f_path.split('.')[-1]
        prob = []
        if suffix.lower() in ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'nef', 'raf']:
            #print("Running prediction on : ",f_path)
            im = cv2.imread(f_path)
            if im is None:
                prob = -1
            else:
                prob = im_test(im)

        logging.info('Prob = ' + str(prob))
        prob_list.append(prob)
        print('Prob: ' + str(prob))

    #
    return prob_list


if __name__ == '__main__':
  root_dir = "/content/DeepFake_ArtifactWarper/testdata/Task_2_3/"
  sub_folding = ["evaluation"]
  categories = ['real', "fake"]

  eva_real = []
  eva_fake = []


  for root, dirs, files in os.walk(root_dir, topdown=True):
    for name in files:
      path = os.path.join(root, name)
      if '.jpg' in path and 'evaluation' in path and 'real' in path:
        eva_real.append(path)
      elif '.jpg' in path and 'evaluation' in path and 'fake' in path:
        eva_fake.append(path)

  print('Real evaluation instances: ', len(eva_real))
  print('Fake evaluation instances: ', len(eva_fake))
  
  RESULTS_REAL = run(eva_real)
  
  print("INFERENCING REAL PREDICTIONS.. \n")
  df = pd.DataFrame({'y_pred': RESULTS_REAL})
  df.to_csv('real.csv', index=False)
  print("INFERENCING FAKE PREDICTIONS.. \n")
  RESULTS_FAKE = run(eva_fake)
  df = pd.DataFrame({'y_pred': RESULTS_FAKE})
  df.to_csv('fake.csv', index=False)

  sess.close()