# DQN_Delay-Sensitive_video_streaming_service

## Background

Online video services account for a large proportion of total data traffic, and the number of dynamic video streaming users is growing rapidly.
We use the Deep Q Network (DQN) algorithm to control the transmission of video chunks in delay-sensitive dynamic video streaming.
In dynamic streaming, a user receives a video stream in chunks, and can simultaneously receive chunks from the latter part of the video while playing the beginning part of the video.




## Setup

pyton <= 3.7
!! This code is written in previous ver. 

```
pip install tensorflow==2.9.1
pip install h5py==3.7.0
pip install keras==2.9.0
pip install matplotlib==3.5.3
pip install numpy==1.21.5
```

## RUN
python DQN.py
