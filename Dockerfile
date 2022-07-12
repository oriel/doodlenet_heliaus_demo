FROM ubuntu:20.04
FROM python:3.8.10

FROM osrf/ros:foxy-desktop


RUN apt-get update -y && \
    apt-get install -y python3-pip python-dev python3-opencv lsb-release curl gnupg2 python3-tk

# # Install ROS2 Foxy
# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
#     echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list

#RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

#RUN sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
# RUN apt-get update -y
# RUN apt-get install -y ros-foxy-desktop
# RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

#RUN apt-get install -y ros-foxy-desktop
#RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc

#RUN ros2 run demo_nodes_cpp talker

#ENTRYPOINT [ "python" ]
#CMD [ "demo_web_no_inference_only_lwir.py" ]

