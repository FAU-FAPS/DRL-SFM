# Nav2 RL controller

## Usage
This nav2 controller plugin implements an action client to use RL algorithms with the Nav2-stack. To use it run an action server which recieves a path as the goal message and returns a velocity command (Twist) as result. E.g. run hunav_rl/path_follower_drlsf or hunav_rl/path_follower_drlvo 
The CalcTwist action defined in the package nav2_rl_controller_msgs is used.

## Acknowledgements
Uses Tutorial code referenced in https://docs.nav2.org/plugin_tutorials/docs/writing_new_nav2controller_plugin.html as a template.