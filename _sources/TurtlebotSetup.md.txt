### Robot Computer Setup
This page details the steps needed to get the computer that is attached to your robot set up correctly. 

#### Dependencies
We use [ROS Indigo Igloo](http://wiki.ros.org/indigo), but any version of ROS after Indigo should work. 
> If your netbook did not come with ROS preinstalled, you will first have to run the following (use the appropriate ROS version for your TurtleBot's Ubuntu distribution):
> ```bash
> sudo apt-get install ros-indigo-desktop-full
> ```
```bash
sudo apt-get install ros-indigo-kobuki libblas-dev liblapack-dev libatlas-base-dev gfortran
sudo pip install future scipy
```

#### Bash Environment Setup
Add the following to your `~/.bashrc` file, which will be run whenever you open a terminal. Make sure to change any paths to match your installation.

```bash
# ROS setup.bash
source /opt/ros/indigo/setup.bash
# Catkin setup.bash
source ~/catkin_ws/devel/setup.bash
# Find your IP address
export ROS_IP=$(hostname -I | awk '{print $1;}')
# Replace 10.0.1.20 with your turtlebot netbook's IP address
export ROS_MASTER_URI=http://10.0.1.20:11311
# ROS needs this sometimes; just in case
export ROS_HOSTNAME=$ROS_IP
```

#### Astra Camera
We use an Astra brand camera. If your camera is different, you may skip this section.
First, clone the `ros_astra_camera` and `ros_astra_launch` repos into `~/catkin_ws/src` as follows: 
```bash
cd ~/catkin_ws/src
git clone https://github.com/orbbec/ros_astra_camera
git clone https://github.com/orbbec/ros_astra_launch
```
Then make the astra_camera package and create the udev rules.
```bash
cd ~/catkin_ws
catkin_make --pkg astra_camera -DFILTER=OFF
source devel/setup.bash
roscd astra_camera && ./scripts/create_udev_rules
```

Finally, to use the camera call:
```bash
roslaunch astra_launch astra.launch
```

#### Start ROS
```bash
# as per your install
source /opt/ros/indigo/setup.bash

# set up networking 
export ROS_IP=$(hostname -I | awk '{print $1;}')
export ROS_HOSTNAME=$ROS_IP
export ROS_MASTER_URI=http://$ROS_HOSTNAME:11311

# as per your install
source $HOME/catkin_ws/devel/setup.bash

# start the ROS master node
roscore &

# start the turtlebot
roslaunch turtlebot_bringup minimal.launch &

# start the camera
roslaunch astra_launch astra.launch &
```
#### [Optional] Automatically start ROS on Netbook Boot

##### Turn on automatic login in the system settings:
In the top right, click the `power gear icon`. Then go to `System Settings` and click `User Accounts` in the bottom right. Click `unlock` in the top right of the window if the options are locked, then click `Automatic Login` so that the slider icon reads "on".

##### Edit the login script
In your favorite text editor, open `~/.profile` or your preferred boot script. At the end, add the following:
```bash
if ! ([ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ]); then  

  # as per your install
  source /opt/ros/indigo/setup.bash

  # sleep to allow network manager to start up
  sleep 5

  # get the IP of this machine
  export ROS_IP=$(hostname -I | awk '{print $1;}')
  export ROS_HOSTNAME=$ROS_IP
  export ROS_MASTER_URI=http://$ROS_HOSTNAME:11311

  # as per your install
  source $HOME/catkin_ws/devel/setup.bash

  roscore &
  # wait for roscore to start so we can start
  # other ros packages
  sleep 10 
  roslaunch turtlebot_bringup minimal.launch &
  roslaunch astra_launch astra.launch &
fi

```


#### [Optional] Network Hotspot
If you don't have a wireless network set up, you can create a hotspot originating from the turtlebot netbook. Note: you will not be able to access the internet while connected to this network.

We followed the instructions [here](https://askubuntu.com/questions/318973/how-do-i-create-a-wifi-hotspot-sharing-wireless-internet-connection-single-adap/609199#609199) to create a hotspot. Note that you may get an ssh error, in which case [this answer](https://askubuntu.com/questions/30080/how-to-solve-connection-refused-errors-in-ssh-connection) might help you.

Sometimes ROS will not be able to connect your hostname with your IP address. You will need to set the `ROS_IP` variable to your IP address.
```bash
ROS_IP=$(hostname -I | awk '{print $1;}')
```


#### [Optional] Save SSH key on turtlebot netbook
To avoid logging in every time you want to ssh or rsync the netbook, use the following commands:
```bash
# follow prompts to generate your ssh key
ssh-keygen

# might need to modify slightly to include your ssh file
# and the netbook's user/hostname combo
cat ~/.ssh/id_rsa.pub | ssh user@hostname 'cat >> .ssh/authorized_keys'
```
