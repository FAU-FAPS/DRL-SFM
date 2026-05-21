#include "nav2_rl_controller/rl_controller.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>

#include "nav2_core/exceptions.hpp"
#include "nav2_rl_controller_msgs/action/calc_twist.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav2_util/node_utils.hpp"
#include "rclcpp/executors.hpp"
#include "rclcpp_action/client_goal_handle.hpp"

using nav2_util::declare_parameter_if_not_declared;
using nav2_util::geometry_utils::euclidean_distance;
using std::abs;
using std::hypot;
using std::max;
using std::min;

namespace nav2_rl_controller
{

using CalcTwist = nav2_rl_controller_msgs::action::CalcTwist;
using GoalHandleCalcTwist = rclcpp_action::ClientGoalHandle<CalcTwist>;

void RLController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, std::string name,
  const std::shared_ptr<tf2_ros::Buffer> tf,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;

  auto node = node_.lock();

  costmap_ros_ = costmap_ros;
  tf_ = tf;
  plugin_name_ = name;
  logger_ = node->get_logger();
  clock_ = node->get_clock();

  declare_parameter_if_not_declared(
    node, plugin_name_ + ".transform_tolerance", rclcpp::ParameterValue(0.1));

  double transform_tolerance;
  node->get_parameter(plugin_name_ + ".transform_tolerance", transform_tolerance);
  transform_tolerance_ = rclcpp::Duration::from_seconds(transform_tolerance);

  global_pub_ = node->create_publisher<nav_msgs::msg::Path>("received_global_plan", 1);

  // Create the action client for the python server:
  action_client_ = rclcpp_action::create_client<CalcTwist>(node, "calc_twist");

  send_goal_options_.goal_response_callback = [this](GoalHandle::SharedPtr goal_handle) {
    if (!goal_handle) {
      RCLCPP_ERROR(logger_, "Goal was rejected by server");
    } else {
      RCLCPP_INFO(logger_, "Goal accepted by server");
      current_goal_handle_ = goal_handle;
    }
  };
  send_goal_options_.result_callback = [this](const GoalHandle::WrappedResult & result) {
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
      RCLCPP_INFO(logger_, "Goal succeeded");
      cmd_vel_.twist = result.result->command;
      new_result_ = true;
    } else {
      RCLCPP_ERROR(logger_, "Goal failed or canceled");
    }
    current_goal_handle_.reset();
    goal_sent_ = false;
  };

  send_goal_options_.feedback_callback =
    [this](GoalHandle::SharedPtr, const std::shared_ptr<const CalcTwist::Feedback> feedback) {
      // This callback intentionally left empty.
      (void)feedback;
    };
}

void RLController::cleanup()
{
  RCLCPP_INFO(
    logger_, "Cleaning up controller: %s of type rl_controller::RLController",
    plugin_name_.c_str());
  global_pub_.reset();
}

void RLController::activate()
{
  RCLCPP_INFO(
    logger_, "Activating controller: %s of type rl_controller::RLController\"  %s",
    plugin_name_.c_str(), plugin_name_.c_str());
  global_pub_->on_activate();

  while (!action_client_->wait_for_action_server(std::chrono::seconds(1))) {
    RCLCPP_INFO(logger_, "Waiting for the action server to be available...");
  }
  RCLCPP_INFO(logger_, "Action server is available.");
}

void RLController::deactivate()
{
  RCLCPP_INFO(
    logger_, "Dectivating controller: %s of type rl_controller::RLController\"  %s",
    plugin_name_.c_str(), plugin_name_.c_str());
  global_pub_->on_deactivate();
}

void RLController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  (void)speed_limit;
  (void)percentage;
}

geometry_msgs::msg::TwistStamped RLController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{
  (void)pose;
  (void)velocity;
  (void)goal_checker;
  // Check if there is still an old goal
  if (current_goal_handle_) {
    RCLCPP_INFO(logger_, "Cancelling old goal");
    action_client_->async_cancel_goal(current_goal_handle_);
    current_goal_handle_.reset();
    goal_sent_ = false;
  }
  new_result_ = false;
  // Call action server
  if (!goal_sent_) {
    RCLCPP_INFO(logger_, "Sending goal to action server");
    auto goal_msg = CalcTwist::Goal();
    goal_msg.path = global_plan_;
    // Send the goal to the action server
    action_client_->async_send_goal(goal_msg, send_goal_options_);
    goal_sent_ = true;
  }
  // loop until a result is received
  while (!new_result_) {
    // spin some time to allow the action server to process the goal
    rclcpp::sleep_for(std::chrono::milliseconds(10));
  }
  // return result
  RCLCPP_INFO(logger_, "Received result from action server");
  new_result_ = false;
  return cmd_vel_;
}

void RLController::setPlan(const nav_msgs::msg::Path & path)
{
  global_pub_->publish(path);
  global_plan_ = path;
}

}  // namespace nav2_rl_controller

// Register this controller as a nav2_core plugin
PLUGINLIB_EXPORT_CLASS(nav2_rl_controller::RLController, nav2_core::Controller)