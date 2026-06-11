import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("my_robot_vision")
    gazebo_share = FindPackageShare("gazebo_ros")

    default_world_path = PathJoinSubstitution([package_share, "worlds", "autonomous_map_3d.world"])
    world_path = LaunchConfiguration("world")
    route_config = LaunchConfiguration("route_config")
    map_image = LaunchConfiguration("map_image")
    gui = LaunchConfiguration("gui")
    camera_viewer = LaunchConfiguration("camera_viewer")
    reverse_direction = LaunchConfiguration("reverse_direction")
    branch_probability = LaunchConfiguration("branch_probability")
    route_seed = LaunchConfiguration("route_seed")
    route_explore_enabled = LaunchConfiguration("route_explore_enabled")
    route_rejoin_enabled = LaunchConfiguration("route_rejoin_enabled")
    route_rejoin_command_error_m = LaunchConfiguration("route_rejoin_command_error_m")
    initial_pose_reset_enabled = LaunchConfiguration("initial_pose_reset_enabled")
    route_offset_x_m = LaunchConfiguration("route_offset_x_m")
    route_offset_y_m = LaunchConfiguration("route_offset_y_m")
    wheel_control_max_angular_rps = LaunchConfiguration("wheel_control_max_angular_rps")
    sign_detection_enabled = LaunchConfiguration("sign_detection_enabled")
    parking_auto_route_enabled = LaunchConfiguration("parking_auto_route_enabled")
    traffic_light_cycle_enabled = LaunchConfiguration("traffic_light_cycle_enabled")
    traffic_light_visual_update_enabled = LaunchConfiguration("traffic_light_visual_update_enabled")
    traffic_light_camera_enabled = LaunchConfiguration("traffic_light_camera_enabled")
    model_path = PathJoinSubstitution([package_share, "models"])
    gazebo_launch = PathJoinSubstitution([gazebo_share, "launch", "gazebo.launch.py"])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world",
                default_value=default_world_path,
                description="Gazebo world file to load.",
            ),
            DeclareLaunchArgument(
                "route_config",
                default_value="",
                description="Optional route graph JSON override. Empty uses the package default.",
            ),
            DeclareLaunchArgument(
                "map_image",
                default_value="",
                description="Optional map image override for route mask generation. Empty uses the package default.",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Start the Gazebo GUI client.",
            ),
            DeclareLaunchArgument(
                "camera_viewer",
                default_value="true",
                description="Start the OpenCV camera viewer window.",
            ),
            DeclareLaunchArgument(
                "reverse_direction",
                default_value="false",
                description="Drive the route in reverse order. Mirror-X map defaults to false.",
            ),
            DeclareLaunchArgument(
                "branch_probability",
                default_value="0.0",
                description="Probability of taking a random inner branch when exploration is disabled.",
            ),
            DeclareLaunchArgument(
                "route_seed",
                default_value="7",
                description="Random seed for branch choices.",
            ),
            DeclareLaunchArgument(
                "route_explore_enabled",
                default_value="false",
                description="Prefer least-visited route branches to exercise all reachable configured paths.",
            ),
            DeclareLaunchArgument(
                "route_rejoin_command_error_m",
                default_value="0.30",
                description="Command/actual position error that triggers route rejoin recovery.",
            ),
            DeclareLaunchArgument(
                "route_rejoin_enabled",
                default_value="false",
                description="Allow route rejoin recovery to snap the car back to the route.",
            ),
            DeclareLaunchArgument(
                "initial_pose_reset_enabled",
                default_value="true",
                description="Allow one startup pose reset to the configured route start.",
            ),
            DeclareLaunchArgument(
                "route_offset_x_m",
                default_value="0.0",
                description="Route centerline X offset in Gazebo model coordinates.",
            ),
            DeclareLaunchArgument(
                "route_offset_y_m",
                default_value="0.0",
                description="Route centerline Y offset in Gazebo model coordinates.",
            ),
            DeclareLaunchArgument(
                "wheel_control_max_angular_rps",
                default_value="2.80",
                description="Maximum angular velocity for the wheel controller.",
            ),
            DeclareLaunchArgument(
                "sign_detection_enabled",
                default_value="false",
                description="Enable camera road-sign detection.",
            ),
            DeclareLaunchArgument(
                "parking_auto_route_enabled",
                default_value="false",
                description="Allow a detected parking sign to auto-plan a dynamic route into the parking area.",
            ),
            DeclareLaunchArgument(
                "traffic_light_cycle_enabled",
                default_value="false",
                description="Enable the simulated traffic light cycle.",
            ),
            DeclareLaunchArgument(
                "traffic_light_visual_update_enabled",
                default_value="false",
                description="Move/update simulated traffic light visual markers.",
            ),
            DeclareLaunchArgument(
                "traffic_light_camera_enabled",
                default_value="false",
                description="Enable camera traffic-light detection.",
            ),
            SetEnvironmentVariable(
                "GAZEBO_MODEL_PATH",
                [model_path, os.pathsep, EnvironmentVariable("GAZEBO_MODEL_PATH", default_value="")],
            ),
            SetEnvironmentVariable("MY_ROBOT_ROUTE_GRAPH_CONFIG", route_config),
            SetEnvironmentVariable("MY_ROBOT_MAP_IMAGE", map_image),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(gazebo_launch),
                launch_arguments={"world": world_path, "gui": gui}.items(),
            ),
            Node(
                package="my_robot_vision",
                executable="gazebo_track_car",
                name="gazebo_track_car",
                output="screen",
                parameters=[
                    {
                        "entity_name": "track_preview_car",
                        "speed": 0.28,
                        "z": 0.02,
                        "yaw_offset": 0.0,
                        "use_sim_time": True,
                        "reverse_direction": ParameterValue(reverse_direction, value_type=bool),
                        "branch_probability": ParameterValue(branch_probability, value_type=float),
                        "route_seed": ParameterValue(route_seed, value_type=int),
                        "route_explore_enabled": ParameterValue(route_explore_enabled, value_type=bool),
                        "start_distance_m": 0.0,
                        "clearance_px": 12,
                        "route_offset_x_m": ParameterValue(route_offset_x_m, value_type=float),
                        "route_offset_y_m": ParameterValue(route_offset_y_m, value_type=float),
                        "sign_detection_enabled": ParameterValue(sign_detection_enabled, value_type=bool),
                        "sign_detection_max_distance_m": 3.4,
                        "sign_detection_fov_rad": 1.95,
                        "sign_detection_min_forward_m": 0.08,
                        "speed_limit_mps": 0.14,
                        "parking_auto_route_enabled": ParameterValue(
                            parking_auto_route_enabled,
                            value_type=bool,
                        ),
                        "traffic_light_state": "green",
                        "traffic_light_cycle_enabled": ParameterValue(traffic_light_cycle_enabled, value_type=bool),
                        "traffic_light_green_duration_s": 5.0,
                        "traffic_light_yellow_duration_s": 1.0,
                        "traffic_light_red_duration_s": 8.0,
                        "traffic_light_trigger_distance_m": 1.70,
                        "traffic_light_stop_distance_m": 0.85,
                        "traffic_light_min_stop_s": 2.0,
                        "traffic_light_visual_update_enabled": ParameterValue(
                            traffic_light_visual_update_enabled,
                            value_type=bool,
                        ),
                        "traffic_light_light_service": "/set_light_properties",
                        "traffic_light_camera_enabled": ParameterValue(traffic_light_camera_enabled, value_type=bool),
                        "traffic_light_camera_timeout_s": 0.25,
                        "drive_wheels": True,
                        "wheel_control_lookahead_m": 0.32,
                        "wheel_control_parking_lookahead_m": 0.20,
                        "wheel_control_heading_gain": 3.8,
                        "wheel_control_max_angular_rps": ParameterValue(
                            wheel_control_max_angular_rps,
                            value_type=float,
                        ),
                        "wheel_control_turn_slowdown_rad": 1.0,
                        "wheel_control_min_speed_scale": 0.35,
                        "wheel_control_terminal_slowdown_m": 0.28,
                        "wheel_control_rotate_heading_error_rad": 1.05,
                        "guided_yaw_rate_limit": 1.8,
                        "guided_yaw_accel_limit": 4.0,
                        "telemetry_period": 2.0,
                        "jitter_log_period": 1.0,
                        "motion_log_verbose": False,
                        "jitter_xy_step_delta_m": 0.006,
                        "jitter_command_error_m": 0.05,
                        "jitter_z_delta_m": 0.0005,
                        "jitter_tilt_rad": 0.003,
                        "jitter_speed_delta_mps": 0.03,
                        "jitter_yaw_step_rad": 0.025,
                        "jitter_yaw_rate_delta_rps": 0.40,
                        "route_rejoin_enabled": ParameterValue(route_rejoin_enabled, value_type=bool),
                        "route_rejoin_outer_only": True,
                        "route_rejoin_command_error_m": ParameterValue(route_rejoin_command_error_m, value_type=float),
                        "route_rejoin_max_route_distance_m": 0.55,
                        "route_rejoin_cooldown_s": 0.20,
                        "initial_pose_reset_enabled": ParameterValue(initial_pose_reset_enabled, value_type=bool),
                        "initial_pose_reset_position_tolerance_m": 0.12,
                        "initial_pose_reset_yaw_tolerance_rad": 0.35,
                    }
                ],
            ),
            Node(
                package="my_robot_vision",
                executable="gazebo_camera_viewer",
                name="gazebo_camera_viewer",
                output="screen",
                condition=IfCondition(camera_viewer),
                parameters=[
                    {
                        "image_topic": "/track_preview_car/camera/image_raw",
                        "window_name": "Track Car Camera",
                        "display_width": 760,
                        "display_height": 428,
                        "use_sim_time": True,
                    }
                ],
            ),
        ]
    )
