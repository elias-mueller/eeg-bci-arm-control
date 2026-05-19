#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "eeg_bci_interfaces/msg/intent.hpp"
#include "rcl_interfaces/msg/floating_point_range.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace
{

constexpr double kDefaultConfidenceThreshold = 0.55;
constexpr double kDefaultJointVelocityRadS = 0.35;
constexpr double kDefaultJointLimitRad = 1.2;
constexpr double kDefaultPublishRateHz = 30.0;
constexpr double kDefaultIntentTimeoutSec = 0.3;

rcl_interfaces::msg::ParameterDescriptor make_read_only_descriptor(
  const std::string & description)
{
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.description = description;
  descriptor.read_only = true;
  return descriptor;
}

rcl_interfaces::msg::ParameterDescriptor make_double_range_descriptor(
  const std::string & description,
  const double from_value,
  const double to_value)
{
  rcl_interfaces::msg::FloatingPointRange range;
  range.from_value = from_value;
  range.to_value = to_value;
  range.step = 0.0;

  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.description = description;
  descriptor.read_only = true;
  descriptor.floating_point_range.push_back(range);
  return descriptor;
}

void validate_positive_parameter(const std::string & name, const double value)
{
  if (!(value > 0.0)) {
    throw std::invalid_argument(name + " must be greater than 0.0");
  }
}

}  // namespace

class IntentJointStateDriver : public rclcpp::Node
{
public:
  IntentJointStateDriver()
  : Node("intent_joint_state_driver"),
    last_publish_time_(this->now())
  {
    const auto intent_topic =
      this->declare_parameter<std::string>(
      "intent_topic",
      "/bci/intent",
      make_read_only_descriptor("Intent topic to subscribe to."));
    const auto joint_state_topic =
      this->declare_parameter<std::string>(
      "joint_state_topic",
      "/joint_states",
      make_read_only_descriptor("JointState topic to publish to."));
    const auto driven_joint_name =
      this->declare_parameter<std::string>(
      "driven_joint_name",
      "panda_joint2",
      make_read_only_descriptor("Published Panda joint to drive from BCI intents."));
    confidence_threshold_ =
      this->declare_parameter<double>(
      "confidence_threshold",
      confidence_threshold_,
      make_double_range_descriptor("Minimum intent confidence accepted for motion.", 0.0, 1.0));
    joint_velocity_rad_s_ =
      this->declare_parameter<double>(
      "joint_velocity_rad_s",
      joint_velocity_rad_s_,
      make_double_range_descriptor(
        "Driven joint velocity for left/right intents in rad/s; must be greater than 0.0.",
        0.0,
        std::numeric_limits<double>::max()));
    joint_limit_rad_ =
      this->declare_parameter<double>(
      "joint_limit_rad",
      joint_limit_rad_,
      make_double_range_descriptor(
        "Symmetric driven joint position limit in radians.",
        0.0,
        std::numeric_limits<double>::max()));
    publish_rate_hz_ =
      this->declare_parameter<double>(
      "publish_rate_hz",
      publish_rate_hz_,
      make_double_range_descriptor(
        "JointState publish rate in Hz.",
        1.0,
        std::numeric_limits<double>::max()));
    intent_timeout_sec_ =
      this->declare_parameter<double>(
      "intent_timeout_sec",
      intent_timeout_sec_,
      make_double_range_descriptor(
        "Seconds to keep the last movement intent before commanding a hold.",
        0.0,
        std::numeric_limits<double>::max()));

    validate_positive_parameter("joint_velocity_rad_s", joint_velocity_rad_s_);
    validate_positive_parameter("intent_timeout_sec", intent_timeout_sec_);

    const auto driven_joint =
      std::find(joint_names_.begin(), joint_names_.end(), driven_joint_name);
    if (driven_joint == joint_names_.end()) {
      throw std::invalid_argument("driven_joint_name must match one of the published Panda joints");
    }
    driven_joint_index_ = static_cast<std::size_t>(driven_joint - joint_names_.begin());

    publisher_ = this->create_publisher<sensor_msgs::msg::JointState>(joint_state_topic, 10);
    subscription_ = this->create_subscription<eeg_bci_interfaces::msg::Intent>(
      intent_topic,
      10,
      [this](const eeg_bci_interfaces::msg::Intent::SharedPtr intent) {
        this->handle_intent(*intent);
      });

    timer_ = this->create_timer(
      std::chrono::duration<double>(1.0 / publish_rate_hz_),
      [this]() {
        this->publish_joint_state();
      });

    RCLCPP_INFO(
      this->get_logger(),
      "Driving %s from %s to %s",
      joint_names_[driven_joint_index_].c_str(),
      intent_topic.c_str(),
      joint_state_topic.c_str());
  }

private:
  void handle_intent(const eeg_bci_interfaces::msg::Intent & intent)
  {
    double commanded_velocity_rad_s = 0.0;
    if (intent.confidence < confidence_threshold_ || intent.label == "rest") {
      commanded_velocity_rad_s = 0.0;
    } else if (intent.label == "left_hand") {
      commanded_velocity_rad_s = -joint_velocity_rad_s_;
    } else if (intent.label == "right_hand") {
      commanded_velocity_rad_s = joint_velocity_rad_s_;
    } else {
      commanded_velocity_rad_s = 0.0;
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        5000,
        "Ignoring unknown intent label '%s'",
        intent.label.c_str());
    }

    commanded_velocity_rad_s_.store(commanded_velocity_rad_s, std::memory_order_relaxed);
    last_intent_time_ns_.store(this->now().nanoseconds(), std::memory_order_relaxed);
  }

  void publish_joint_state()
  {
    const auto now = this->now();
    const auto dt = std::clamp((now - last_publish_time_).seconds(), 0.0, 2.0 / publish_rate_hz_);
    last_publish_time_ = now;
    auto commanded_velocity_rad_s = commanded_velocity_rad_s_.load(std::memory_order_relaxed);
    const auto last_intent_time_ns =
      last_intent_time_ns_.load(std::memory_order_relaxed);
    const auto seconds_since_intent =
      static_cast<double>(now.nanoseconds() - last_intent_time_ns) / 1e9;
    if (seconds_since_intent > intent_timeout_sec_) {
      commanded_velocity_rad_s = 0.0;
    }

    joint_positions_[driven_joint_index_] = std::clamp(
      joint_positions_[driven_joint_index_] + commanded_velocity_rad_s * dt,
      -joint_limit_rad_,
      joint_limit_rad_);

    sensor_msgs::msg::JointState joint_state;
    joint_state.header.stamp = now;
    joint_state.name = joint_names_;
    joint_state.position.assign(joint_positions_.begin(), joint_positions_.end());
    publisher_->publish(joint_state);
  }

  // Keep joint_names_ in sync with the revolute joints in urdf/panda_visual.urdf.
  std::vector<std::string> joint_names_{
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
  };
  std::array<double, 7> joint_positions_{};
  std::size_t driven_joint_index_{1};
  double confidence_threshold_{kDefaultConfidenceThreshold};
  double joint_velocity_rad_s_{kDefaultJointVelocityRadS};
  double joint_limit_rad_{kDefaultJointLimitRad};
  double publish_rate_hz_{kDefaultPublishRateHz};
  double intent_timeout_sec_{kDefaultIntentTimeoutSec};
  std::atomic<double> commanded_velocity_rad_s_{0.0};
  std::atomic<int64_t> last_intent_time_ns_{0};
  rclcpp::Time last_publish_time_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_{};
  rclcpp::Subscription<eeg_bci_interfaces::msg::Intent>::SharedPtr subscription_{};
  rclcpp::TimerBase::SharedPtr timer_{};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IntentJointStateDriver>());
  rclcpp::shutdown();
  return 0;
}
