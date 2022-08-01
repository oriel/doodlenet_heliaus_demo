# Jetson / PC demo

# Docker configuration

Docker should be configured to be able to access the GPU throught nvidia-containers. 
To install nvidia-container-runtime:
```
sudo apt-get install nvidia-container-runtime
```
If docker is not yet configured to use nvidia-container-runtime, it can be done with:

```
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
```

Then restart docker service:

```
 sudo systemctl daemon-reload
 sudo systemctl restart docker
``` 

# Building and installation instructions

# 0) Clone the private git repo

```
git clone https://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@github.com/oriel/doodlenet_heliaus_demo.git
```

# 1) Build docker image

Go to the path of the cloned repository:

```
cd doodlenet_heliaus_demo
```

## a) Build for PC

```
docker build -f Dockerfile_pc_ros2 -t "doodlenet_ros2:beta" .
```

## b) Build for jetson (will only work if building from a jetson - L4T 32.6.1 [ JetPack 4.6 ])

```
docker build -f Dockerfile_jetson_ros2  -t "doodlenet_ros2:beta" .
```

# 2) Run docker image:

Run this bash script to allow gpu usage, host network, X graphic server:

```
bash runDocker.sh
```

# 3) Running ros2 demo inside docker image

```
python3 autosens_demo.py
```

# 3.1) Running ros2 demo with custom ROS2 topics for input cameras

Note that the demo with default parameters use "source_images_rgb" and "source_images_lwir" as input ROS2 topics.
You may use custom input topics for the Autosens demo with:

```
python3 autosens_demo.py --rgb_topic /camera/flea/left/aligned --lwir_topic /camera/smartIR640/cam_0008/ir_frame_enhanced/recified
```

You may check other available customisable parameters with:

```
python3 autosens_demo.py --help
```

