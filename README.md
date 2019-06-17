# int-event-fusion

This repository is associated with "Event-driven Video Frame Synthesis" ([arXiv](https://arxiv.org/abs/1902.09680)).

# Abstract
Video frame synthesis is an active computer vision problem which has applications in video compression, streaming, editing, and understanding. In this work, we present a computational high speed video synthesis framework. Our framework takes as inputs two types of data streams: an intensity frame stream and a neuromorphic event stream, which consists of asynchronous bipolar "events" which encode brightness variations over time at over 1000 fps. We introduce an algorithm to recover a space-time video from these two modes of observations. We factor the reconstruction into a physical model-based reconstruction (PBR) process and a residual denoising process. We use a differentiable model to approximate the physical sensing process, which enables stochastic gradient descent optimization using automatic differentiation. Residual errors in PBR reconstruction are further reduced by training a residual denoiser to remove reconstruction artifacts. The video output from our reconstruction algorithm has both high frame rate and well-recovered spatial features. Our framework is capable of handling challenging scenes that include fast motion and strong occlusions.


# Organization of this repository (in branch 'win10')
'demo' contains notebook for illustrating our proposed differentiable model for fusing event frames and intensity frame(s).
'devel' contains code in development
'sample_data' contains sample data (in .npy format) for the [Need-for-Speed](http://ci2cv.net/nfs/index.html) dataset and the [DAVIS-240](http://rpg.ifi.uzh.ch/davis_data.html) dataset.
'sample_preparation' contains scripts to split downloaded data, prepare training/testing samples.

# Datasets

We use a mixture of the [Adobe-240](https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/#dataset) and the [Need for Speed](http://ci2cv.net/nfs/index.html) dataset is used for training the residual denoiser. 
The [DAVIS](http://rpg.ifi.uzh.ch/davis_data.html) dataset is used for evaluating our differentiable model based reconstruction.

# Environment tested
Windows10
anaconda3
cuda 9.0
cudnn 7.0

tensorflow-gpu 1.12
# Dependencies
scikit-image
keras with tensorflow backend
moviepy (optional, mainly used for generating gif files)
