# Jetson / PC demo

# 0) Clone the private git repo

```sh
git clone https://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@github.com/oriel/doodlenet_heliaus_demo.git
```

# 1) Build docker image

## a) Build for PC

```
docker build -f Dockerfile_pc_ros2 -t "doodlenet_ros2:pc" .
```

## b) Build for jetson (will only work if building from a jetson)

```
docker build -f Dockerfile_jetson_ros2  -t "doodlenet_ros2:jetson" .
```

# 2) Run docker image (run this bash script to allow gpu usage, host network, X graphic server):

```
bash runDocker.sh
```

# 3) Running ros2 demo inside docker image

```
python3 test_heliaus_doodlenet_ros2.py
```

# 3.1) Running ros2 demo with custom ROS2 topics for input cameras

Note that the demo with default parameters use "source_images_rgb" and "source_images_lwir" as input ROS2 topics.
You may use custom input topics with:

```
python3 test_heliaus_doodlenet_ros2.py --rgb_topic custom_rgb_cam --lwir_topic custom_lwir_cam 
```

You may check other available customisable parameters with:

```
python3 test_heliaus_doodlenet_ros2.py --help
```

