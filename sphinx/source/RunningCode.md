### Running the code

You have two options when it comes to running your code on the TurtleBot from your laptop:
- Option 1: Run the code on your local computer while connected to the TurtleBot's network ('critterbot2' in the RLAI lab).
- Option 2: Run on the TurtleBot's netbook after 'rsync'ing the code from your computer to the turtle bot.

#### Pros and cons
- Option 1 is heavily affected by network quality, and in our tests had highly variable latency.
- Option 2 requires using the TurtleBot's netbook which may not be as powerful as you like (in that case consider replacing it, or tethering the turtlebot to a desktop computer).

#### Option 1: Run code on your local computer

1. Ensure you are connected to the TurtleBot's network.

2. Ensure Desktop-Full ROS is installed. Follow these instructions to do so (if you have Ubuntu 16.04! Otherwise find the corresponding ROS version that is pertinent to your Linux distro): http://wiki.ros.org/kinetic/Installation/Ubuntu. Note the note under Step 1.3 if you are having trouble with it.

3. Ensure your environment variable ROS_MASTER_URI is set properly. Use command 'export ROS_MASTER_URI=http://<turtlebot's ip>:11311' and replace <turtlebot's ip> with the turtlebot IP.

4. Run the python file you want on your laptop \[e.g. Use command 'python wall_demo.py'\].

#### Option 2: Run code on the TurtleBot's netbook

1. Ensure you are connected to the TurtleBot's network.

2. Open a terminal and ssh into your TurtleBot [Use 'ssh turtlebot@<turtlebot's ip>'

4. In a new terminal (non ssh) transfer all files to the TurtleBot. \[e.g. 'rsync -r /path/to/script_folder/ /path/from/home/my_folder\]. Make sure there is a slash at the end of the source path but not at the end of the destination path.

5. Inside the ssh terminal run your code \[e.g. 'python my_folder/wall_demo.py'\].
