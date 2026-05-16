cmake_minimum_required(VERSION 3.8)
project(manipulator_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(eeg_bci_interfaces REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(intent_command_logger src/intent_command_logger.cpp)
ament_target_dependencies(intent_command_logger
  eeg_bci_interfaces
  rclcpp
)

install(TARGETS intent_command_logger
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
