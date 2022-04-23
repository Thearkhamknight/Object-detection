# Object-detection
This Tensorflow Object Detection API is the original creation of Yuki Takahashi, with the github username Yuki678.
I merely modified a few scripts to make it more compatible so it can be run more smoothly.
The original README file created by Mr. Takahashi is in driving-object-detection.

Also inside Mr. Takahashi's driving-object-detection repository is a requirements text file and a setup environment script 'setup_env.sh'.
 Many of the scripts used to run this notebook are copyrighted by the Tensorflow authors and licensed under the Apache License, Version 2.0 (the "License").
A copy of the license can be obtained at 
http://www.apache.org/licenses/LICENSE-2.0

## Installation:
The installation is for this Object Detection API is fairly straightforward. Make sure you have Python and Jupyter Notebook installed in your work space. This Object Detection notebook will also clone the driving-object-detection and Ex_Scripts repositories to your work space so ensure that you have enough memory. One thing I would like to note is that the terminal commands are for the Debian OS so you will have to change some of them to fit your OS. After cloning the driving-object-detection repository, open the README.MD file within that repository for more information on how to add additional models, converting the exported model to earlier versions of Tensorflow, and increase inference speed.


## Instructions:
This notebook uses transfer learning, using models pretrained on the Coco dataset. 
If you would like to use your own images then after cloning driving-object-detection, navigate to driving-object-detection/images, delete all the images of traffic lights and the xml files. 
Upload your own images. Then label them in the object-detection bounding box format using LabelImg in PASCAL VOC format and upload their respective xml files. Make sure the xml files have the exact same name as the image files, with the exception of the file extension. Ensure that the bounding box xml annotation files are in PASCAL VOC format, as this is necessary for the model to be trained properly. The notebook will partition the images into a training and test set, convert the xml files in each set into a single csv file, and finally convert the csv file into a tf.record file for training and evaluation. 

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
In my workspace the configuration file is under Object-detection/driving-object-detection/models/tf2/my_centernet_resnet50_v1_fpn. The folder after tf2 will depend on what model you choose to train on.
This is what my pipeline.config file is, I will specify what changes to make inside the configuration file. After making these changes, make sure to save the pipeline.config file.

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

  batch_size: 16
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
  max_number_of_boxes: 3
  unpad_groundtruth_tensors: false

  fine_tune_checkpoint_version: V2 # In fine_tune_checkpoint below, specify the full checkpoint file path. 
  # Object-detection/models/research/pretrained_model/checkpoint/ckpt-0 should stay the same but include any preceding directories.
  fine_tune_checkpoint: "/home/faizan_samad/testing/Object-detection/models/research/pretrained_model/checkpoint/ckpt-0" #Specify the full file path.
  fine_tune_checkpoint_type: "detection" #Make sure the checkpoint type is detection
}

train_input_reader: {
  label_map_path: "/home/faizan_samad/testing/Object-detection/driving-object-detection/annotations/label_map.pbtxt" #Specify the full file path. Change preceding directories before Object-detection.
  tf_record_input_reader {
    input_path:  "/home/faizan_samad/testing/Object-detection/driving-object-detection/annotations/train.record" #Specify the full file path. Change preceding directories before Object-detection.
  }
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
  batch_size: 1;
}

eval_input_reader: {
  label_map_path: "/home/faizan_samad/testing/Object-detection/driving-object-detection/annotations/label_map.pbtxt" #Specify the full file path. Change preceding directories before Object-detection.
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "/home/faizan_samad/testing/Object-detection/driving-object-detection/annotations/test.record" #Specify the full file path. Change preceding directories before Object-detection.
  }
}
```


