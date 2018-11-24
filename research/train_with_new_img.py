# This script runs all processes from unzipping new data to training:

# before running: copy files form airsim output to cfd
# scp ~/code/models/research/object_detection/data/airSim/

# 1. unzip image file
# 2.  rm -r old o-airSim
# 3. filename_changer.py
# 4. create_obst_tf_record.py
# 5. train

import zipfile
import os
import sys
import shutil


# traindir suffix from input
train_dir_suffix = sys.argv[1] #'+str(train_dir_suffix)+'


#unzip images in i-airSim.zip
with zipfile.ZipFile("object_detection/data/airSim/i-airSim.zip","r") as zip_ref:
    zip_ref.extractall("object_detection/data/airSim/i-airSim/")
    print('unzip')

# filename_changer.py
os.system('python object_detection/dataset_tools/filename_changer.py')
print('data format changed')

# create_obst_tf_record.py
os.system('python object_detection/dataset_tools/create_obst_tf_record.py')
print('train and val.record created')

# train use screen in neccessary
print('start training...')
os.system('python object_detection/legacy/train.py --logtostderr --train_dir=./object_detection/models/train'+str(train_dir_suffix)+' --pipeline_config_path=object_detection/samples/configs/mask_rcnn_inception_resnet_v2_atrous_coco.config')

