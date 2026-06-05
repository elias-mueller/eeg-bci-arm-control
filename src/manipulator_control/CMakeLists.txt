cmake_minimum_required(VERSION 3.8)
project(manipulator_control)

# Emit build/<pkg>/compile_commands.json so editor IntelliSense (MS C/C++ / clangd)
# resolves ROS and generated-interface headers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(eeg_bci_interfaces REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(intent_command_logger src/intent_command_logger.cpp)
target_compile_features(intent_command_logger PRIVATE cxx_std_17)
ament_target_dependencies(intent_command_logger
  eeg_bci_interfaces
  rclcpp
)

add_executable(intent_joint_state_driver src/intent_joint_state_driver.cpp)
target_compile_features(intent_joint_state_driver PRIVATE cxx_std_17)
ament_target_dependencies(intent_joint_state_driver
  eeg_bci_interfaces
  rcl_interfaces
  rclcpp
  sensor_msgs
)

install(TARGETS
  intent_command_logger
  intent_joint_state_driver
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake REQUIRED)

  add_launch_test(test/test_intent_joint_state_driver.py)
endif()

ament_package()
