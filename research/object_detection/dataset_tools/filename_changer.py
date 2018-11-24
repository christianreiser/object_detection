"""
use this script to change filenames from airsim recording to a format that create_obst_tf_record.py can use it.

# split images in scene and segmentation into  different folders
# rename into same samed scene /seg pairs

1. move and rename
2. RGBA to RGB
3. airsim_gt to binary_gt
4. write xml files
5. write trainval.txt list

"""
import os
import numpy as np
import scipy.misc
import PIL.Image
import shutil


#image height and width
width  = 1365
height = 800
drone_color = np.array([16, 154, 4]) #in airsim
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

binary_gt = np.ones((height,width), dtype=np.int8)
binary_gt = binary_gt*3

def airsim_ground_truth_to_binary_ground_truth(img_path):
    #image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
    airsim_gt = scipy.misc.imread(img_path)
    out= np.ones((height,width), dtype=np.int8)*3
    binary_gt = np.all(airsim_gt == drone_color, axis=2)#, out= out)
    binary_gt = binary_gt*1 # bool to int
    scipy.misc.imsave(img_path, binary_gt)
    return binary_gt

# mkdir
if os.path.exists('object_detection/data/airSim/o-airSim/'):
    shutil.rmtree('object_detection/data/airSim/o-airSim/')

if not os.path.exists('object_detection/data/airSim/o-airSim/'):
    os.makedirs('object_detection//data/airSim/o-airSim/')
if not os.path.exists('object_detection/data/airSim/o-airSim/images/'):
    os.makedirs('object_detection/data/airSim/o-airSim/images/')

if not os.path.exists('object_detection/data/airSim/o-airSim/tf-record/'):
    os.makedirs('object_detection/data/airSim/o-airSim/tf-record/')

if not os.path.exists('object_detection/data/airSim/o-airSim/annotations/trimaps/'):
    os.makedirs('object_detection/data/airSim/o-airSim/annotations/trimaps/')

if not os.path.exists('object_detection/data/airSim/o-airSim/annotations/xmls/'):
    os.makedirs('object_detection/data/airSim/o-airSim/annotations/xmls/')



myimages = [] #list of image filenames

dirFiles = os.listdir('object_detection/data/airSim/i-airSim/.') #list of directory files
dirFiles.sort()
sorted(dirFiles) #sort numerically in ascending order

for files in dirFiles: #filter out all non pngs
    if '.png' in files:
        myimages.append(files)



scene, segmentation = split_list(myimages)

if len(scene) != len(segmentation):

    print('len scene != len(seg)')

print('\n',len(scene),'\n\nseg',len(segmentation))

i = 0
for filename in scene:
    dst_scene = str(i) + ".png"
    src_scene = 'object_detection/data/airSim/i-airSim/' + filename
    dst_scene = 'object_detection/data/airSim/o-airSim/images/' + dst_scene

    # rename() function will
    # rename all the files
    os.rename(src_scene, dst_scene)

    # convert airsim images from RGBA to RGB
    rgba_image_scene = PIL.Image.open(dst_scene)
    rgb_image_scene  = rgba_image_scene.convert('RGB')
    # convert from png to jpg
    rgb_image_scene.save('object_detection/data/airSim/o-airSim/images/'+str(i) + ".jpg")
    os.remove('object_detection/data/airSim/o-airSim/images/'+str(i) + ".png")
    i += 1


trainval_file = open('object_detection/data/airSim/o-airSim/annotations/trainval.txt',"w+")

i = 0
for filename in segmentation:
    dst = str(i) + ".png"
    src = 'object_detection/data/airSim/i-airSim/' + filename
    dst = 'object_detection/data/airSim/o-airSim/annotations/trimaps/' + dst

    # rename() function will
    # rename all the files
    os.rename(src, dst)

    # convert airsim images from RGBA to RGB
    rgba_image = PIL.Image.open(dst)
    rgb_image = rgba_image.convert('RGB')
    rgb_image.save(dst)

    # airsim_ground_truth_to_binary_ground_truth
    binary_gt = airsim_ground_truth_to_binary_ground_truth(dst)

    #create xml file for every scene/seg-pair
    xml_file = open('object_detection/data/airSim/o-airSim/annotations/xmls/'+str(i)+".xml","w+")
    xml_file.write("<annotation><filename>"+str(i)+".jpg</filename><size><width>"+str(width)+"</width><height>"+str(height)+"</height><depth>3</depth></size><object><truncated>0</truncated><difficult>0</difficult></object></annotation>")
    xml_file.close()
    print('xml at:','object_detection/data/airSim/o-airSim/annotations/xmls/'+str(i)+".xml")
    #write trainval.txt list
    trainval_file.write(str(i)+"\n")
    print("\n"+str(i))
    i += 1
trainval_file.close()



import matplotlib.pyplot as plt

fig=plt.figure()
img_scene = rgb_image_scene
img_seg   = rgb_image
img_bin   = binary_gt

fig.add_subplot(1,3,1)
plt.imshow(img_scene)

fig.add_subplot(1,3,2)
plt.imshow(img_seg)

fig.add_subplot(1,3,3)
plt.imshow(img_bin)
plt.show()

#from PIL import Image
#im1 = Image.open('object_detection/data/airSim/o-airSim/images/'+str(i-1) + ".jpg")
#im2 = Image.open(dst)
#blended = Image.blend(im1, im2, alpha=0.5)
#img_scene.show()
#blended.save("blended.png")


#img_scene.paste(img_seg, (0, 0), img_seg)
#img_scene.show()


print('done.')
