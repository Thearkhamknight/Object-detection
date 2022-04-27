# Object-detection
This Tensorflow Object Detection API is the original creation of Yuki Takahashi, with the github username Yuki678. I merely modified a few scripts to make 
it more compatible so it can be run more smoothly. For more information see the end of this README.
The objective is, using models pretrained on the Coco Dataset, to use transfer learning with Tensorflow's Object Detection API to train a selected model 
to detect objects in different classes within a set of images. For example, Mr. Takahashi trained this 
model to detect red, yellow, and green traffic lights. I used this model to detect redactions, signatures, initials, and dates from a variety of
documents. Note: When you run this notebook, you will be cloning two other repositories to your local work station, driving-object-detection and 
Ex_Scripts. 

At the end, you will have the trained model exported inside the cloned driving-object-detection repository with the name trained_model.tar.gz. 
The evaluation cell, cell 31, will give you precise metrics on how your model performs. It will tell you the Average Precision and Average Recall 
at different IOU thresholds and for different sizes of the output bounding boxes. Coco defines a small area as between 0 and 32^2 pixels, medium area as 
between 32^2 pixels and 96^2 pixels, and large area as between 96^2 pixels and 1e5^2 pixels. (For more information on IOU and Average Precision, check 
out the links I provide at the end of this README.) Finally, the inference cell, which is the last cell, will visually show you how well your model 
performs on your test images.
The model or detector will output the bounding boxes, the class that each bounding box belongs to, and the detection scores or probabilities for 
each respective bounding box belonging to its class. This information will be visually superimposed on your test set images so you can verify visually
how your model performs. With the information provided by both the evaluation and inference cells, you will be able to analyze how your model performs 
both numerically and visually.




## Installation:
The installation is for this Object Detection API is fairly straightforward. Make sure you have Python and Jupyter Notebook installed in your work space in order to run this notebook. This Object Detection notebook will also clone the driving-object-detection and Ex_Scripts repositories to your work space so ensure that you have enough memory. One thing I would like to note is that some of the terminal commands are specific to the Debian OS, therefore you will have to change some of them to fit your OS. After cloning the driving-object-detection repository, open the README.MD file within that repository for more information on how to add additional models, convert the exported model to earlier versions of Tensorflow, and increase inference speed.


## Instructions for using your own data:
This notebook uses transfer learning, using models pretrained on the Coco dataset. 
Before cloning the driving-object-detection and Ex_Scripts repositories, you will need to decide which model you would like to use. This occurs in cell 5.
Here is the link for additional Tensorflow Object-Detection Models trained on Coco:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

Most of the significant changes you will make will come after cell 18, which is where you clone the driving-object-detection and Ex_Scripts repositories. For your reference, this is what cell 18 is: 
```
# This cell clones all the content from the original driving object detection repository into the repository directory path.
# It also checks that the label map and pipeline files exist.
# I added the line: os.makedirs(model_dir, exist_ok=True). after cloning  and pulling the repositories. 
# This ensures that a training sub folder will be created.
# The original clone command is !git clone {repo_url}
# Since the repository is public you will not be prompted to give your github username and password.
# But if you are prompted here is what you should do.

# Instead of !git clone {repo_url} type the command:
# !git clone https://<Username>:<Password>@github.com/yuki678/driving-object-detection.git
# Change the directory to fit your workspace.
import os
%cd ./
# Clean up

!rm -rf {repo_dir_path}

# Clone

!git clone https://github.com/Thearkhamknight/driving-object-detection.git
!git clone https://github.com/Thearkhamknight/Ex_Scripts.git
# !git clone {repo_url} 
# Pull (just in case the repo already exists)
%cd {repo_dir_path}
!git pull

# Check if label map and pipeline files exist
assert os.path.isfile(label_map_pbtxt_fname), '`{}` not exist'.format(label_map_pbtxt_fname)
assert os.path.isfile(pipeline_fname), '`{}` not exist'.format(pipeline_fname)
os.makedirs(model_dir, exist_ok=True)
```

To train this model on your own images, after cloning the driving-object-detection repository navigate to driving-object-detection/images and 
delete all the images of traffic lights and the xml files. 
Upload your own images to driving-object-detection/images. Then label them in the object-detection bounding box format using LabelImg in PASCAL VOC format 
and upload their respective xml files inside the same directory. Make sure the xml files have the exact same name as the image files, with the exception 
of the file extension. Ensure that the bounding box xml annotation files are in PASCAL VOC format, as this is necessary for the model to be trained 
properly. The notebook will partition the images into a training and test set, convert the xml files in each set into a single csv file, and finally 
convert the csv file into a tf.record file for training and evaluation. 

Here is a link on how to install LabelImg https://github.com/tzutalin/labelImg#installation.

Next go inside driving-object-detection/annotations and change the label_map.pbtxt file to fit your dataset. 
For example, if you are trying to detect animals you would change it to:

```
item {
    id: 1
    name: 'dog'
}

item {
    id: 2
    name: 'cat'
}

item {
    id: 3
    name: 'parrot'
}
```
Finally, open the pipeline.config file.
The pipeline.config file is the only file where you will have to specify full absolute paths for your workspace.
In my workspace the configuration file is under Object-detection/driving-object-detection/models/tf2/my_centernet_resnet50_v1_fpn. The folder after tf2 
will depend on what model you choose to train on.
This is what my pipeline.config file is, and I will specify what changes to make inside the configuration file. After making these changes, make sure to 
save the pipeline.config file.

```
# CenterNet meta-architecture from the "Objects as Points" [1] paper
# with the ResNet-v2-101 backbone. The ResNet backbone has a few differences
# as compared to the one mentioned in the paper, hence the performance is
# slightly worse. This config is TPU comptatible.
# [1]: https://arxiv.org/abs/1904.07850
#

model {
  center_net {
    num_classes: 3 #Change this to the same number of classes as label_map.pbtxt
    feature_extractor {
      type: "resnet_v1_50_fpn"
    }
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 512
        max_dimension: 512
        pad_to_max_dimension: true
      }
    }
    object_detection_task {
      task_loss_weight: 1.0
      offset_loss_weight: 1.0
      scale_loss_weight: 0.1
      localization_loss {
        l1_localization_loss {
        }
      }
    }
    object_center_params {
      object_center_loss_weight: 1.0
      min_box_overlap_iou: 0.7
      max_box_predictions: 100
      classification_loss {
        penalty_reduced_logistic_focal_loss {
          alpha: 2.0
          beta: 4.0
        }
      }
    }
  }
}

train_config: {

  batch_size: 16 # Increase/Decrease this value depending on the available memory (Higher values require more memory and vice-versa)
  num_steps: 10000

  data_augmentation_options {
    random_horizontal_flip {
    }
  }

  data_augmentation_options {
    random_crop_image {
      min_aspect_ratio: 0.5
      max_aspect_ratio: 1.7
      random_coef: 0.25
    }
  }


  data_augmentation_options {
    random_adjust_hue {
    }
  }

  data_augmentation_options {
    random_adjust_contrast {
    }
  }

  data_augmentation_options {
    random_adjust_saturation {
    }
  }

  data_augmentation_options {
    random_adjust_brightness {
    }
  }

  data_augmentation_options {
    random_absolute_pad_image {
       max_height_padding: 200
       max_width_padding: 200
       pad_color: [0, 0, 0]
    }
  }

  optimizer {
    adam_optimizer: {
      epsilon: 1e-7  # Match tf.keras.optimizers.Adam's default.
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: 1e-3
          total_steps: 10000
          warmup_learning_rate: 2.5e-4
          warmup_steps: 1000
        }
      }
    }
    use_moving_average: false
  }
  max_number_of_boxes: 3 # This will have to match the maximum number of bounding boxes identified in a single image within the your entire image set. 
  #For example, if you identified that the maximum number of bounding boxes is 20, change this value to 20.
  unpad_groundtruth_tensors: false

  fine_tune_checkpoint_version: V2 # In fine_tune_checkpoint below, specify the full checkpoint file path. 
  # Object-detection/models/research/pretrained_model/checkpoint/ckpt-0 should stay the same but include any preceding directories.
  fine_tune_checkpoint: "/home/faizan_samad/testing/Object-detection/models/research/pretrained_model/checkpoint/ckpt-0" #Specify the full file path.
  fine_tune_checkpoint_type: "detection" #Make sure the checkpoint type is detection
}

train_input_reader: {
  label_map_path: "/home/faizan_samad/testing/Object-detection/driving-object-detection/annotations/label_map.pbtxt" #Specify the full file path. 
  #Change preceding directories before Object-detection.
  tf_record_input_reader {
    input_path:  "/home/faizan_samad/testing/Object-detection/driving-object-detection/annotations/train.record" #Specify the full file path. 
    #Change preceding directories before Object-detection.
  }
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
  batch_size: 1;
}

eval_input_reader: {
  label_map_path: "/home/faizan_samad/testing/Object-detection/driving-object-detection/annotations/label_map.pbtxt" #Specify the full file path. 
  #Change preceding directories before Object-detection.
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "/home/faizan_samad/testing/Object-detection/driving-object-detection/annotations/test.record" #Specify the full file path. 
    #Change preceding directories before Object-detection.
  }
}
```
## Credit to Yuki Takahashi:

The original README file created by Mr. Takahashi is in driving-object-detection.
Also inside Mr. Takahashi's driving-object-detection repository is a requirements text file and a setup environment script 'setup_env.sh'.
 Many of the scripts used to run this notebook are copyrighted by the Tensorflow authors and licensed under the Apache License, Version 2.0 (the 
 "License").
A copy of the license can be obtained at 
http://www.apache.org/licenses/LICENSE-2.0


## Links for Average Precision and IOU:

https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52

https://towardsdatascience.com/map-mean-average-precision-might-confuse-you-5956f1bfa9e2#:~:text=The%20mean%20Average%20Precision%20or,an%20IoU%20threshold%20of%200.5.
