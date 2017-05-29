### Astra Camera
First, clone the `ros_astra_camera` and `ros_astra_launch` repos into `~/catkin_ws/src` as follows:
```
cd ~/catkin_ws/src
git clone https://github.com/orbbec/ros_astra_camera
git clone https://github.com/orbbec/ros_astra_launch
```
Then make the astra_camera package and create the udev rules.
```
cd ~/catkin_ws
catkin_make --pkg astra_camera -DFILTER=OFF
roscd astra_camera && ./scripts/create_udev_rules
```

Finally, to use the camera call:
```
source ~/catkin_ws/devel/setup.bash
roslaunch astra_launch astra.launch
```

### Automatically start ROS on Netbook Boot

#### Turn on automatic login in the system settings:
In the top right, click the `power gear icon`. Then go to `System Settings` and click `User Accounts` in the bottom right. Click `unlock` in the top right of the window if the options are locked, then click `Automatic Login` so that the slider icon reads "on".

#### Edit the login script
In your favorite text editor, open `~/.profile` or your preffered login script. Paste into it the following:
```{bash}
if ! ([ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ]); then  

  # as per your install
  source /opt/ros/indigo/setup.bash

  # get the IP of this machine
  export ROS_IP=$(hostname -I | awk '{print $1;}')
  export ROS_HOSTNAME=$ROS_IP
  export ROS_MASTER_URI=http://$ROS_HOSTNAME:11311

  # as per your install
  source ~/catkin_ws/devel/setup.bash

  roscore &
  roslaunch turtlebot_bringup minimal.launch &
  roslaunch astra_launch astra.launch &
fi

```


### (Optional) Hospot/AP Networking
Since we don't have a wireless network set up, we use an access point (ap) network originating from the turtlebot netbook. Note: you will not be able to access the internet while connected to this network.

We followed the instructions [here](https://askubuntu.com/questions/180733/how-to-setup-an-access-point-mode-wi-fi-hotspot/180734#180734) (Pay attention to step 3.1!).

Sometimes ROS will not be able to connect your hostname with your IP address. You will need to set the `ROS_IP` variable to your IP address.
```{bash}
ROS_IP=$(hostname -I | awk '{print $1;}')
```
