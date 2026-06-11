from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    show_debug = LaunchConfiguration("show_debug")
    show_gui = LaunchConfiguration("show_gui")

    return LaunchDescription(
        [
            DeclareLaunchArgument("show_debug", default_value="true"),
            DeclareLaunchArgument("show_gui", default_value="true"),
            Node(
                package="my_robot_vision",
                executable="robot_sim",
                name="robot_sim",
                output="screen",
                parameters=[{"show_gui": show_gui}],
            ),
            Node(
                package="my_robot_vision",
                executable="virtual_camera",
                name="virtual_camera",
                output="screen",
            ),
            Node(
                package="my_robot_vision",
                executable="line_follower",
                name="line_follower",
                output="screen",
                parameters=[{"show_debug": show_debug}],
            ),
        ]
    )
