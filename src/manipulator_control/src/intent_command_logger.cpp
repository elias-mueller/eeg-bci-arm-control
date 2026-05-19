#include <memory>
#include <string>

#include "eeg_bci_interfaces/msg/intent.hpp"
#include "rclcpp/rclcpp.hpp"

class IntentCommandLogger : public rclcpp::Node
{
public:
  IntentCommandLogger()
  : Node("intent_command_logger")
  {
    const auto topic = this->declare_parameter<std::string>("intent_topic", "/bci/intent");
    subscription_ = this->create_subscription<eeg_bci_interfaces::msg::Intent>(
      topic,
      10,
      [this](const eeg_bci_interfaces::msg::Intent::SharedPtr intent) {
        RCLCPP_INFO(
          this->get_logger(),
          "Received intent '%s' with confidence %.3f",
          intent->label.c_str(),
          intent->confidence);
      });

    RCLCPP_INFO(this->get_logger(), "Listening for BCI intents on %s", topic.c_str());
  }

private:
  rclcpp::Subscription<eeg_bci_interfaces::msg::Intent>::SharedPtr subscription_{};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IntentCommandLogger>());
  rclcpp::shutdown();
  return 0;
}
