Instructions here

### (Optional) Hospot/AP Networking
Since we don't have a wireless network set up, we use an access point (ap) network originating from the turtlebot netbook. Note: you will not be able to access the internet while connected to this network.

We followed the instructions [here](https://askubuntu.com/questions/180733/how-to-setup-an-access-point-mode-wi-fi-hotspot/180734#180734) (Pay attention to step 3.1!).

Sometimes ROS will not be able to connect your hostname with your IP address. You will need to set the `ROS_IP` variable to your IP address.
```{bash}
ROS_IP=$(hostname -I | awk '{print $1;}')
```
