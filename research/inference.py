from __future__ import print_function
import os
import sys
import shutil

# get train_dir_suffix from input
train_dir_suffix = sys.argv[1] #'+str(train_dir_suffix)+' 
num_img = sys.argv[2] #'+str(num_img)+' 

# print all ckpt numbers
path = 'object_detection/models/train'+str(train_dir_suffix)+'/'
files = os.listdir(path)
for name in files:
    if name[:11] == 'model.ckpt-':
        print(name[11:17])
        #print([int(s) for s in name[11:17].split() if s.isdigit()])

# get ckpt number from input
ckpt_num =  input('enter ckpt-num')


"""
# del old fine_tuned_model_chrei
if os.path.exists('object_detection/models/train'+str(train_dir_suffix)+'/fine_tuned_model_chrei/'):
    shutil.rmtree('object_detection/models/train'+str(train_dir_suffix)+'/fine_tuned_model_chrei/')

# export inference graph
os.system('python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ./object_detection/samples/configs/mask_rcnn_inception_resnet_v2_atrous_coco.config --trained_checkpoint_prefix ./object_detection/models/train'+str(train_dir_suffix)+'/model.ckpt-'+str(ckpt_num)+' --output_directory ./object_detection/models/train'+str(train_dir_suffix)+'/fine_tuned_model_chrei/')


"""
# infere and display
os.system('python object_detection/seg_test.py '+str(train_dir_suffix)+' '+str(num_img)+' && display object_detection/models/train'+str(train_dir_suffix)+'/fine_tuned_model_chrei/result1.png')

