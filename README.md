# lost_semi_auto_pipes
This repo is currently under construction!

## About
Adaptions of pylessons repo for tiny yolo pre- and postprocessing (https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3) 
and Triton Inference Server (https://github.com/triton-inference-server)

## Description
Pipelines for semi automatic annotation.
To get faster annotation data these pipelines use pretrained models for classification or object detection  proposals.
It is possible to train a model in a loop or if there is a trained model for your data you can generate new annotations.
The object detection pipelines work with a [tiny yolo v4](git@github.com:l3p-cv/lost_yolov3_tf2.git) Tensorflow model.

## Installation
1. Import Pipelines
    * to import the pipelines to your LOST- Application, follow the [README](https://github.com/l3p-cv/lost-pipeline-zoo/blob/master/README.md) in LOST Pipeline Zoo
    * at step 6. add the url of this [repository](git@github.com:l3p-cv/lost_semi_auto_pipes.git)

2. Install model server and -repository
    * [install Triton Inference Server](https://github.com/triton-inference-server/server.git)
    * create a [model repository](https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md)
    * structure the [model repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md)

    Example of a Tensorflow model repository structure:
    ```
    model_repository
    ├── model_name_1
    │   ├── 1
    │   │   └── model.savedmodel
    │   │       ├── assets
    │   │       ├── variables
    │   │       │   ├── variables.data-00000-of-00001
    │   │       │   └── variables.index
    │   │       ├── keras_metadata.pb
    │   │       └── saved_model.pb
    │   ├── 2
    │   │   └── model.savedmodel
    │   │       ├── assets
    │   │       ├── variables
    │   │       │   ├── variables.data-00000-of-00001
    │   │       │   └── variables.index
    │   │       ├── keras_metadata.pb
    │   │       └── saved_model.pb
    │   ├── config.pbtxt
    │   └── model_classess.txt
    └── model_name_2
        └── 1
            └── model.savedmodel
                ├── assets
                ├── variables
                │   ├── variables.data-00000-of-00001
                │   └── variables.index
                ├── keras_metadata.pb
                └── saved_model.pb

    ```
    * The directory named by numbers presents the model version. In default settings the pipeline will use the newest version.
    * labels.txt represents the dictionary for the annotation classes. It has to fit to the loaded model.

## request_triton_mia
### Description
Request multi image annotations with a Tensorflow model classification proposals for all images of a specified data source.

### How do I use the pipeline ?
1. load Tensorflow model to the model repository
    * Add the model_classes.txt (model classes have to fit to your model).
    * It is possible to config specific settings by your own in the config.pbtxt (for example load all model versions: version_policy: { all: {}}) 

2. start and configure the pipeline
    * start the pipeline request_triton_mia
    * choose your image data directory in datasource
    * at the script block, click on "arguments available" and configure them
        * model_name: Enter the name of your model. It is the name of the directory in your model repository, see structure example above.
        * model_version: Enter the model version. The default is newest version
        * batch_size: Size of the image batch for model input. The maximum depends on the model.
        * url: Enter the url of the Triton Inference Server
        * port: Enter the port of the Triton Inference Server (at the moment only 8000 for http request is possible)


## request_triton_mia_loop
### Description
Request multi image annotations with a for all images of a specified data source.

### How do I use the pipeline ?

## tiny_yolo_triton_sia
### Description
Request single image annotations with model object detection proposals for all images of a specified data source.
Supported is **tiny yolo v4 Tensorflow** model.
The model is trainable in a loop with the tiny_yolo_triton_sia_loop pipeline.

### How do I use the pipeline ?
1. load Tensorflow model to the model repository
    * It is possible to config specific settings by your own in the config.pbtxt.

2. start and configure the pipeline
    * start the pipeline tiny_yolo_triton_sia
    * there are two datasources
        * choose your image data directory in one datasource
        * choose your anno data directory in the other datasource. In this directory has to be an anno data file as json.
    * at the script block, click on "arguments available" and configure them
        * valid_imgtypes: These are the supported image types
        * model_name: Enter the name of your model. It is the name of the directory in your model repository, see structure example above.
        * url: Enter the url of the Triton Inference Server
        * port: Enter the port of the Triton Inference Server (at the moment only port 8000 for http request is possible)

## tiny_yolo_triton_sia_loop
### Description
Request single image annotations with model object detection proposals for all images of a specified data source.
This Pipeline runs in a loop. At the first iteration the images will be annotated manual. Use this annotation data to [train](git@github.com:l3p-cv/lost_yolov3_tf2.git) the model
for next iteration. The number of loops is pending on the number of images in the datasource and the image batch size per loop.

Supported is **tiny yolo v4 Tensorflow** model.

### How do I use the pipeline ?
1. load Tensorflow model to the model repository
    * It is possible to config specific settings by your own in the config.pbtxt.

2. start and configure the pipeline
    * start the pipeline tiny_yolo_triton_sia_loop
    * choose your image data directory in datasource
    * at the script block, click on "arguments available" and configure them
        * valid_imgtypes: These are the supported image types
        * model_name: Enter the name of your model. It is the name of the directory in your model repository, see structure example above.
        * url: Enter the url of the Triton Inference Server
        * port: Enter the port of the Triton Inference Server (at the moment only port 8000 for http request is possible)
