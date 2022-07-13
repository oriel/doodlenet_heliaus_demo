# Jetson / PC demo

# 0) Clone the private git repo

```sh
git clone https://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@github.com/oriel/doodlenet_heliaus_demo.git
```

# 1) Build docker image

## a) Build for jetson

```sh
docker build -f Dockerfile_jetson_ros2  -t "doodlenet_ros2:jetson" .
```

## a) Build for PC

```sh
docker build -f Dockerfile_ros2_pc -t "doodlenet_ros2:pc" .
```

# 2) Run docker image (run this bash script to allow gpu usage, host network, X graphic server):
```sh
bash runDocker.sh
```
# 3) Running ros2 demo inside docker image
```sh
python3 test_heliaus_doodlenet_ros2.py
```

