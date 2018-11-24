# Common Commands and Errors/Fixes
## ablage:
scp -r chrei@volocopter.org@cfd01:~/code/models/research/object_detection/samples/configs/

scp -r chrei@...
ssh pc-dell-lnx-24.volocopter.org -X
ssh pc-mobil-30.volocopter.org -X
ssh chrei@volocopter.org@cfd01

----------
## common commands
### if new in terminal

ssh chrei@volocopter.org@cfd01 -X
cd && cd code/ && source ./venv/bin/activate && cd models/research/ && export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 

------------
### extract frames from mp4

ffmpeg -i Video.mp4 Pictures%d.jpg

bring all training img and labels to

/home/chrei/code/christianInternship/cvLaptop/labelImg/data/images/train

----------------
### labelimg

python /home/chrei/code/christianInternship/cvLaptop/labelImg/labelImg.py

----------
### from airsim to api
#### scp
scp  ~/Desktop/images/* chrei@volocopter.org@cfd01:~/code/models/research/object_detection/data/airSim/i-airSim/

#### filename changer
python object_detection/dataset_tools/filename_changer.py


----------------
### generate_tfrecord.py

#### mask
python object_detection/dataset_tools/create_obst_tf_record.py

#### box
cd && cd code/christianInternship/cvLaptop/labelImg/data/
python3 generate_tfrecord.py --csv_input=csv/train_labels.csv --output_path=tfrecord/train.record
python3 generate_tfrecord.py --csv_input=csv/eval_labels.csv --output_path=tfrecord/eval.record

cd code/christianInternship/cvLaptop/labelImg/data/
mv tfrecord/train.record ~/code/models/research/object_detection/samples/configs/training/data/
mv tfrecord/eval.record ~/code/models/research/object_detection/samples/configs/training/data/

---------

### train

cd && cd code/models/research/object_detection/samples/configs/training

#### box
python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=pipeline.config

#### mask (change train_dir)
python object_detection/legacy/train.py --logtostderr --train_dir=./object_detection/models/train2 --pipeline_config_path=object_detection/samples/configs/mask_rcnn_inception_v2_obs/t.config

-------------

### TensorBoard (change logdir)

tensorboard --logdir=object_detection/models/train/
------------------------

### export_inference_graph. Info: change [#]

#### check chechkpoint (dir)
vi object_detection/models/train

#### export box
python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./pipeline.config --trained_checkpoint_prefix ./models/train/model.ckpt-332 --output_directory ./fine_tuned_model

#### expport mask (change train2x and ckpt)
python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ./object_detection/samples/configs/mask_rcnn_inception_resnet_v2_atrous_coco.config --trained_checkpoint_prefix ./object_detection/models/train4/model.ckpt-126 --output_directory ./object_detection/models/train4/fine_tuned_model_chrei/

### infere mask and display (change2x)
python object_detection/seg_test.py 4 && display object_detection/models/train4/fine_tuned_model_chrei/result1.png




----------------

### eval

cd && cd code/models/research/

#### box
python object_detection/legacy/eval.py --logtostderr --pipeline_config_path=object_detection/samples/configs/training/pipeline.config     --checkpoint_dir=object_detection/samples/configs/training/fine_tuned_model     --eval_dir=object_detection/samples/configs/training/eval

#### mask train (traindir
python object_detection/legacy/eval.py --logtostderr --pipeline_config_path=object_detection/samples/configs/mask_rcnn_inception_resnet_v2_atrous_coco.config --checkpoint_dir=object_detection/models/train8/fine_tuned_model_chrei --eval_dir=object_detection/models/eval



-----------------------

### run test 

ssh pc-dell-lnx-24.volocopter.org -X

cd && cd code/ && source ./venv/bin/activate && cd models/research/ && export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim && cd object_detection/ && python object_detection_tutorial.py

scp chrei@pc-dell-lnx-24:~/code/models/research/object_detection/samples/configs/training/fine_tuned_model/result* ~/code/

visualization_utils.py

vi object_detection/utils/visualization_utils.py
 
l:561 visualize_boxes_and_labels_on_image_array

---------------








## common errors:
### general:
0. not using tensorflow-gpu due to cuda version proplems

### training 
1. ModuleNotFoundError: No module named 'object_detection': new in terminal?
2. ImportError: cannot import name 'trainer' or preprocessor_pb2: installation instructions or/and copy trainer.py to research/object_detection
3.  ValueError: No variables to save ->check folder structue
	object_detection/samples/configs/training
					-> .config
					-> data / train.record
					-> models/(3times from model-zoo).ckpt 
					-> models/train
4. ValueError('First step cannot be zero.'): change step: 0 to step: 1 in .config
5. AttributeError: module 'object_detection.utils.dataset_util' has no attribute 'make_initializable_iterator':
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/issues/91
change in train.py "dataset_util" to dataset_builder"
5. InvalidArgumentError: Invalid JPEG data or crop window, data size: find corrupt image by size info and delete it
6. InvalidArgumentError (see above for traceback): Incompatible shapes: [4,1917] vs. [50,1] -> remove ssd_random_crop in pip.config

NotFoundError (see above for traceback): Key FirstStageFeatureExtractor/cell_0/1x1/weights not found in checkpoint
	 [[Node: save/RestoreV2 = RestoreV2[dtypes=[DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT64], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save/Const_0_0, save/RestoreV2/tensor_names, save/RestoreV2/shape_and_slices)]]

### evaluation
1.     NameError: name 'unicode' is not defined
		in models/research/object_detection/utils/object_detection_evaluation.py"
		replace this: unicode(
		to this: str(
2. ValueError: Image with id b'166.jpg' already added. -> update 'num_examples:' in config file





