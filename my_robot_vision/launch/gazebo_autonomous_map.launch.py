from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("my_robot_vision")
    gazebo_share = FindPackageShare("gazebo_ros")

    world_path = PathJoinSubstitution([package_share, "worlds", "autonomous_map_3d.world"])
    model_path = PathJoinSubstitution([package_share, "models"])
    gazebo_launch = PathJoinSubstitution([gazebo_share, "launch", "gazebo.launch.py"])

    return LaunchDescription(
        [
            SetEnvironmentVariable("GAZEBO_MODEL_PATH", model_path),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(gazebo_launch),
                launch_arguments={"world": world_path}.items(),
            ),
        ]
    )
