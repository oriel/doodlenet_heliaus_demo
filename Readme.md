# Jetson demo

# 1) Build docker image

## a) Build for jetson

docker build -f Dockerfile_jetson_ros2  -t "doodlenet_ros2:jetson" .

## a) Build for PC

docker build -f Dockerfile_ros2_pc -t "doodlenet_ros2:pc" .

# 2) Run docker image

docker run -it doodlenet_ros2_jetson:Dockerfile_jetson_ros2

# 3) Running ros2 demo inside docker image

python3 test_heliaus_doodlenet_ros2.py

# Training: benchmarking models on MF dataset

- Training:

python train_segmentation_reproducible.py

- Evaluation:

python run_evaluation_test.py

- Instalation:

pip install virtualenv  
# virtualenv venv (deprecated)
python3 -m venv venv
source venv/bin/activate  
pip install -r requirements.txt  
  
python train.py --help  
python train.py --dataset kaist_rgbt --use_trains  

