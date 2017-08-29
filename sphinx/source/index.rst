.. Robot Learning Project documentation master file, created by
   sphinx-quickstart on Wed Aug  2 12:11:49 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. toctree::
   :maxdepth: 2
   :hidden:

   Code Overview <Overview>
   Robot Computer Setup <TurtlebotSetup>
   Running the Code <RunningCode>
   File Index <pyrst/modules>

Welcome
========

The Robot Learning Project is an initiative from the University of Alberta with the goal of creating agents to learn arbitrary information about the environment using low-level sensorimotor data. Learning directly from sensorimotor data is fundamental to the creation of robots that can perform complex tasks without human intervention. By using robotic platforms, researchers can address many obstacles that may be difficult to replicate in a simulation, such as the randomness in delays between sensing stimuli and the environment changing because of the robotic agent's actions.

Unfortunately, robotic learning platforms are often difficult to set up because of the hardware challenges that reinforcement learning researchers may not be used to facing. The Robot Learning Project aims to provide a launchpad for researchers to set up their own robotic learning platform, quickly set up experiments, and focus on reinforcement learning research rather than technical difficulties.

Getting Started
---------------
To get started, first follow the :doc:`Robot Computer Setup <TurtlebotSetup>` instructions.

You also need to decide if the computer that will run the learning code should be physically attached to the robot. If not, the learning computer will have to communicate with the robot computer over a wireless network. See :doc:`Running the code <RunningCode>` for pros and cons to each setup.

.. note::
   **Wirelessly connected learning computer setup:**

   Dependencies::

      sudo apt-get install ros-indigo-desktop-full ros-indigo-kobuki libblas-dev liblapack-dev libatlas-base-dev gfortran
      sudo pip install future scipy

   You will also have to set some environment variables on the wirelessly connected computer. Add the following to your ``~/.bashrc``, while making sure to change any paths to match your installation.::

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

   and start ROS::

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

On the learning computer -- whether directly or wirelessly connected to the robot -- clone the repo::

   git clone https://github.com/AmiiThinks/RobotLearning.git
   cd RobotLearning

Finally, call the :py:mod:`wall_demo_example` file::

   python src/wall_demo_example.py

Easy data plotting can be done using ROS's `rqt_plot <http://wiki.ros.org/rqt_plot>`_.
