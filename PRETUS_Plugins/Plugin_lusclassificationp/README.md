# B-line classification for LUS - pytorch version

Authors: 

Hamideh Kerdegari (hamideh.kerdegari@kcl.ac.uk)
Nhat Phung (nhat.phung@kcl.ac.uk)
Alberto Gomez (alberto.gomez@kcl.ac.uk)

# Summary

This plug-in classifies input videos into 5 classes: A-lines, B-lines, confluent B-line, Consilidation, and Pleural effussion. The model is implemented in Pytorch.

# Usage

## Usage within PRETUS
After building the standalone software [PRETUS](https://github.com/gomezalberto/pretus), and after adding the path where this plug-in is installed in the pretus config file (`~/.config/iFIND/PRETUS.conf`), you shopuld see the plug-in and it's help when running pretus:

```bash
$ ./launcher_pretus.sh -h

...
(13) Plugin Name: 'LUS Classification P'

# PLUGIN LUS Classification
   Automatic Classification in 5 classes.
	--lusclassificationp_stream <val> [ type: STRING]	Name of the stream(s) that this plug-in takes as input. (Default: ) 
	--lusclassificationp_layer <val> [ type: INT]	Number of the input layer to pass to the processing task. If negative, starts 
                                             		from te end. (Default: 0) 
	--lusclassificationp_framerate <val> [ type: FLOAT]	Frame rate at which the plugin does the work. (Default: 20) 
	--lusclassificationp_verbose <val> [ type: BOOL]	Whether to print debug information (1) or not (0). (Default: 0) 
	--lusclassificationp_time <val> [ type: BOOL]	Whether to measure execution time (1) or not (0). (Default: 0) 
	--lusclassificationp_showimage <val> [ type: INT]	Whether to display realtime image outputs in the central window (1) or not (0). 
                                                 		(Default: <1 for input plugins, 0 for the rest>) 
	--lusclassificationp_showwidget <val> [ type: INT]	Whether to display widget with plugin information (1-4) or not (0). Location is 
                                                  		1- top left, 2- top right, 3-bottom left, 4-bottom right. (Default: visible, 
                                                  		default location depends on widget.) 
   Plugin-specific arguments:
	--lusclassificationp_modelname <*.h5> [ type: STRING]	Model file name (without folder). (Default: models/model_001) 
	--lusclassificationp_nframes <val> [ type: INT]	Number of frames in the buffer. (Default: 30) 
	--lusclassificationp_cropbounds xmin:ymin:width:height [ type: STRING]	set of four colon-delimited numbers with the pixels to define the crop bounds 
                                                                      		(Default: 0.25:0.1:0.55:0.75) 
	--lusclassificationp_abscropbounds 0/1 [ type: BOOL]	whether the crop bounds are provided in relative values (0 - in %) or absolute 
                                                    		(1 -in pixels) (Default: 0) 
	--lusclassificationp_showassistant 0/1 [ type: BOOL]	whether to show the AI assistant (1) or not (0) (Default: 1) 
	--lusclassificationp_output <filename> [ type: STRING]	path to the output filename where results will be written (Default: output.txt) 

```

To run the plug-in, you need to specify a video (or real time input). The results will be saved into the file specified by `--lusclassification_output`. An example call would be:

```bash
$ ./launcher_pretus.sh -pipeline "videomanager>lusclassificationp>gui" --videomanager_input ~/data/VITAL/lungs/01NVb-003-004-1lus.mp4 --lusclassification_modelname models/model_002 --videomanager_loop 1 --lusclassificationp_nframes 5 --lusclassificationp_showassistant 1 --lusclassification_output ~/data/VITAL/lungs/01NVb-003-004-1lus-output.txt

```

Which produces a session similar to the one shown in the figure below:

![pretus](art/pretus-lus.gif)



# Build instructions

## Dependencies

The minimum requirements are:


* VTK. You need to fill in the VTK
* ITK (for video inputs, built with the `ITKVideoBridgeOpencv` option `ON`).  You need to fill in the VTK
* Boost
* Qt 5 (tested with 5.12). You need to fill in the `QT_DIR` variable in CMake
* c++11
* Python (you need the python libraries, include dirs)

Additionally, for this plug-in: 

* Python 3 (tested on Python 3.7) 
* Python 3 library
* [PyBind11](https://pybind11.readthedocs.io/en/stable/advanced/cast/overview.html) (for the python interface if required), with python 3.
* [matplotlib]()
* [numpy]()
* [opencv-python]()
* [scikit-learn]()
* [scipy]()
* [torch]() 1.14.0


The python include and binary should be the same used for pybind11. For example, if the python distribution comes from Anaconda, your `PYTHON_INCLUDE_DIR` in the CMake will be something like `<HOME_FOLDER>/anaconda3/include/python3.7m` and your `PYTHON_LIBRARY` will be something like `<HOME_FOLDER>/anaconda3/lib/libpython3.7m.so`.

## Build and install


Launch CMake configure and generate. Then make and install
``` bash
$ make && make install
$ export  PYTHONPATH=<your selected install folder\>:"$PYTHONPATH"
```
And launch.

## Troubleshooting

* You will need to set `CMAKE_INSTALL_PREFIX` to a path where your python scripts will go, e.g. <your selected install folder\> .
* Make sure that `PYTHON_INCLUDE_DIR` and `PYTHON_LIBRARY` are correctly set to Python 3.

