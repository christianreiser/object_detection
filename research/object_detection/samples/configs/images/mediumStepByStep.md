# Part 2: Converting Existing Dataset to TFRecord

def class_text_to_int(row_label):
    if row_label == 'black':
        return 1
    if row_label == 'white':
        return 2
    elif row_label == 'red':
        return 3
    else:
        None

#Folder structure:
Copy the object_detection/samples/configs/(...).config into training-folder.
In training-folder create a folder called data and move your TFRecord file inside of it. 
Create another folder called models and move the .ckpt (checkpoint) files (3 of them) of the pre-trained model you selected into this folder.
Inside the models folder create another folder called train.


#Modifying the Config File:
- number of classes
- num_steps
- num_examples ->number of evaluation samples

- fine_tune_checkpoint: "models/model.ckpt"
- input_path -> TFRecord file
- label_map_path -> .pbtxt (item {id:1 name: 'Red'})
