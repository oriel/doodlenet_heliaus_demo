# Jetson demo

# 1) Build docker image

## a) Build for jetson

docker build -f Dockerfile_jetson_ros2  -t "doodlenet_ros2:jetson" .

## a) Build for PC

docker build -f Dockerfile_ros2_pc -t "doodlenet_ros2:pc" .

# 2) Run docker image (run this bash script to allow gpu usage, host network, X graphic server):

bash runDocker.sh

# 3) Running ros2 demo inside docker image

python3 test_heliaus_doodlenet_ros2.py


