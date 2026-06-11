import os
import re
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


COPY_ROOT = "/home/quocbao/my_robot_vision_lane_pid_project"
DEFAULT_WORLD = f"{COPY_ROOT}/mirrored_world_lab/generated/map_2.world"
DEFAULT_ROUTE_CONFIG = (
    f"{COPY_ROOT}/route_drawing_lab/outputs/session_20260608_132843/"
    "autonomous_route_graph_flat_magenta_original2d_parking_stub_stop.json"
)
CAMERA_RAY_VISUAL_NAMES = {
    "camera_view_left_ray_visual",
    "camera_view_center_ray_visual",
    "camera_view_right_ray_visual",
}


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def prepare_experiment_world(context, *args, **kwargs):
    source_world = Path(LaunchConfiguration("world").perform(context)).expanduser()
    if not truthy(LaunchConfiguration("hide_camera_rays").perform(context)):
        return [SetLaunchConfiguration("effective_world", str(source_world))]

    output_dir = Path(COPY_ROOT) / "lane_pid_experiment_runtime"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_world = output_dir / f"{source_world.stem}_no_camera_rays.world"

    world_text = source_world.read_text(encoding="utf-8")
    modified = False
    for visual_name in CAMERA_RAY_VISUAL_NAMES:
        pattern = re.compile(
            rf"(<visual name=['\"]{re.escape(visual_name)}['\"]>.*?</visual>)",
            re.DOTALL,
        )

        def shrink_visual(match):
            nonlocal modified
            modified = True
            block = match.group(1)
            return re.sub(r"<size>[^<]+</size>", "<size>0.001 0.001 0.001</size>", block, count=1)

        world_text = pattern.sub(shrink_visual, world_text, count=1)

    if modified:
        output_world.write_text(world_text, encoding="utf-8")
        return [SetLaunchConfiguration("effective_world", str(output_world))]
    return [SetLaunchConfiguration("effective_world", str(source_world))]


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("my_robot_vision")
    gazebo_share = FindPackageShare("gazebo_ros")

    effective_world = LaunchConfiguration("effective_world")
    gui = LaunchConfiguration("gui")
    route_config = LaunchConfiguration("route_config")
    camera_viewer = LaunchConfiguration("camera_viewer")
    raw_camera_viewer = LaunchConfiguration("raw_camera_viewer")
    camera_viewer_show_window = LaunchConfiguration("camera_viewer_show_window")
    sign_detection_enabled = LaunchConfiguration("sign_detection_enabled")
    traffic_light_enabled = LaunchConfiguration("traffic_light_enabled")
    traffic_light_visual_cycle = LaunchConfiguration("traffic_light_visual_cycle")
    route_guard_reverse_direction = LaunchConfiguration("route_guard_reverse_direction")
    initial_pose_reset_enabled = LaunchConfiguration("initial_pose_reset_enabled")
    initial_pose_x_m = LaunchConfiguration("initial_pose_x_m")
    initial_pose_y_m = LaunchConfiguration("initial_pose_y_m")
    initial_pose_yaw_rad = LaunchConfiguration("initial_pose_yaw_rad")
    kp = LaunchConfiguration("kp")
    ki = LaunchConfiguration("ki")
    kd = LaunchConfiguration("kd")
    base_speed = LaunchConfiguration("base_speed")
    min_speed = LaunchConfiguration("min_speed")
    max_angular = LaunchConfiguration("max_angular")
    steering_sign = LaunchConfiguration("steering_sign")
    model_path = PathJoinSubstitution([package_share, "models"])
    gazebo_launch = PathJoinSubstitution([gazebo_share, "launch", "gazebo.launch.py"])

    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value=DEFAULT_WORLD),
            DeclareLaunchArgument("route_config", default_value=DEFAULT_ROUTE_CONFIG),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("camera_viewer", default_value="true"),
            DeclareLaunchArgument("raw_camera_viewer", default_value="true"),
            DeclareLaunchArgument("camera_viewer_show_window", default_value="true"),
            DeclareLaunchArgument("hide_camera_rays", default_value="true"),
            DeclareLaunchArgument("sign_detection_enabled", default_value="true"),
            DeclareLaunchArgument("traffic_light_enabled", default_value="true"),
            DeclareLaunchArgument("traffic_light_visual_cycle", default_value="true"),
            DeclareLaunchArgument("route_guard_reverse_direction", default_value="true"),
            DeclareLaunchArgument("initial_pose_reset_enabled", default_value="true"),
            DeclareLaunchArgument("initial_pose_x_m", default_value="2.1796875"),
            DeclareLaunchArgument("initial_pose_y_m", default_value="1.8046875"),
            DeclareLaunchArgument("initial_pose_yaw_rad", default_value="-1.5707963267948966"),
            DeclareLaunchArgument("kp", default_value="0.005"),
            DeclareLaunchArgument("ki", default_value="0.0"),
            DeclareLaunchArgument("kd", default_value="0.0007"),
            DeclareLaunchArgument("base_speed", default_value="0.22"),
            DeclareLaunchArgument("min_speed", default_value="0.08"),
            DeclareLaunchArgument("max_angular", default_value="1.50"),
            DeclareLaunchArgument("steering_sign", default_value="-1.0"),
            SetEnvironmentVariable(
                "GAZEBO_MODEL_PATH",
                [model_path, os.pathsep, EnvironmentVariable("GAZEBO_MODEL_PATH", default_value="")],
            ),
            OpaqueFunction(function=prepare_experiment_world),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(gazebo_launch),
                launch_arguments={"world": effective_world, "gui": gui}.items(),
            ),
            Node(
                package="my_robot_vision",
                executable="gazebo_traffic_light_visual_cycle",
                name="gazebo_traffic_light_visual_cycle",
                output="screen",
                condition=IfCondition(traffic_light_visual_cycle),
                parameters=[
                    {
                        "route_config": route_config,
                        "green_duration_s": 5.0,
                        "yellow_duration_s": 1.0,
                        "red_duration_s": 8.0,
                        "cycle_enabled": True,
                        "initial_state": "green",
                        "publish_topic": "/lane_pid_experiment/visual_traffic_light_state",
                        "use_sim_time": True,
                    }
                ],
            ),
            Node(
                package="my_robot_vision",
                executable="gazebo_lane_pid_driver",
                name="gazebo_lane_pid_driver",
                output="screen",
                parameters=[
                    {
                        "camera_topic": "/track_preview_car/camera/image_raw",
                        "cmd_topic": "/track_preview_car/cmd_vel",
                        "debug_overlay_topic": "/lane_pid_experiment/debug_overlay",
                        "ipm_topic": "/lane_pid_experiment/ipm",
                        "mask_topic": "/lane_pid_experiment/mask",
                        "lane_debug_topic": "/lane_pid_experiment/lane_debug",
                        "traffic_light_topic": "/lane_pid_experiment/camera_traffic_light_state",
                        "route_config": route_config,
                        "route_guard_enabled": True,
                        "route_guard_reverse_direction": ParameterValue(
                            route_guard_reverse_direction,
                            value_type=bool,
                        ),
                        "route_guard_blend_start_m": 0.06,
                        "route_guard_blend_full_m": 0.28,
                        "route_guard_slow_distance_m": 0.38,
                        "route_guard_stop_distance_m": 0.95,
                        "route_guard_low_confidence_weight": 0.40,
                        "route_guard_single_lane_weight": 0.26,
                        "src_bottom_left_ratio": [0.04, 0.99],
                        "src_bottom_right_ratio": [0.96, 0.99],
                        "src_top_right_ratio": [0.73, 0.44],
                        "src_top_left_ratio": [0.27, 0.44],
                        "dst_margin_ratio": 0.12,
                        "roi_top_ratio": 0.34,
                        "histogram_top_ratio": 0.50,
                        "kp": ParameterValue(kp, value_type=float),
                        "ki": ParameterValue(ki, value_type=float),
                        "kd": ParameterValue(kd, value_type=float),
                        "base_speed": ParameterValue(base_speed, value_type=float),
                        "min_speed": ParameterValue(min_speed, value_type=float),
                        "max_angular": ParameterValue(max_angular, value_type=float),
                        "steering_sign": ParameterValue(steering_sign, value_type=float),
                        "center_smoothing_alpha": 0.28,
                        "max_center_jump_px": 28.0,
                        "hough_pair_blend_weight": 0.08,
                        "hough_single_blend_weight": 0.0,
                        "max_debug_hough_lines": 0,
                        "lane_error_deadband_px": 3.5,
                        "angular_smoothing_alpha": 0.36,
                        "max_angular_delta_per_s": 2.8,
                        "curve_slowdown_min_factor": 0.62,
                        "curve_slowdown_angular_start": 0.45,
                        "initial_pose_reset_enabled": ParameterValue(
                            initial_pose_reset_enabled,
                            value_type=bool,
                        ),
                        "initial_pose_from_route": True,
                        "initial_pose_x_m": ParameterValue(initial_pose_x_m, value_type=float),
                        "initial_pose_y_m": ParameterValue(initial_pose_y_m, value_type=float),
                        "initial_pose_yaw_rad": ParameterValue(initial_pose_yaw_rad, value_type=float),
                        "publish_debug_images": True,
                        "traffic_light_enabled": ParameterValue(traffic_light_enabled, value_type=bool),
                        "traffic_light_proximity_gate_enabled": True,
                        "traffic_light_scene_gate_strict": False,
                        "traffic_light_red_approach_slow_enabled": True,
                        "traffic_light_release_on_non_actionable": True,
                        "traffic_light_lost_release_after_min_stop": True,
                        "traffic_light_stop_min_area_px": 900,
                        "traffic_light_slow_min_area_px": 650,
                        "traffic_light_release_min_area_px": 450,
                        "traffic_light_action_min_height_px": 22,
                        "traffic_light_action_min_bottom_y_ratio": 0.28,
                        "traffic_light_action_min_score": 120.0,
                        "traffic_light_action_stable_frames": 2,
                        "traffic_light_action_max_distance_m": 1.55,
                        "traffic_light_action_min_forward_m": -0.05,
                        "traffic_light_action_fov_rad": 2.20,
                        "traffic_light_min_stop_s": 2.0,
                        "camera_stale_stop_enabled": True,
                        "camera_stale_timeout_s": 0.75,
                        "camera_stale_log_period_s": 1.0,
                        "use_sim_time": True,
                    }
                ],
            ),
            Node(
                package="my_robot_vision",
                executable="gazebo_road_sign_camera_detector",
                name="gazebo_road_sign_camera_detector",
                output="screen",
                condition=IfCondition(sign_detection_enabled),
                parameters=[
                    {
                        "image_topic": "/track_preview_car/camera/image_raw",
                        "publish_topic": "/lane_pid_experiment/camera_sign_detection",
                        "traffic_light_detection_topic": "/lane_pid_experiment/camera_traffic_light_state",
                        "traffic_light_suppression_enabled": True,
                        "roi_top_ratio": 0.08,
                        "roi_bottom_ratio": 0.84,
                        "min_colored_area_px": 55,
                        "min_bbox_height_px": 14,
                        "min_score": 0.30,
                        "action_min_area_px": 1500,
                        "action_min_height_px": 34,
                        "stable_frames": 2,
                        "maneuver_stable_frames": 4,
                        "timeout_s": 0.45,
                        "use_sim_time": True,
                    }
                ],
            ),
            Node(
                package="my_robot_vision",
                executable="gazebo_camera_viewer",
                name="gazebo_lane_pid_debug_viewer",
                output="screen",
                condition=IfCondition(camera_viewer),
                parameters=[
                    {
                        "image_topic": "/lane_pid_experiment/debug_overlay",
                        "window_name": "Lane PID Experiment",
                        "show_window": ParameterValue(camera_viewer_show_window, value_type=bool),
                        "display_width": 760,
                        "display_height": 760,
                        "window_x": 40,
                        "window_y": 40,
                        "sign_overlay_enabled": False,
                        "traffic_light_overlay_enabled": False,
                        "traffic_light_detection_topic": "/lane_pid_experiment/camera_traffic_light_state",
                        "overlay_topic": "/lane_pid_experiment/viewer_overlay",
                        "use_sim_time": True,
                    }
                ],
            ),
            Node(
                package="my_robot_vision",
                executable="gazebo_camera_viewer",
                name="gazebo_lane_pid_raw_camera_viewer",
                output="screen",
                condition=IfCondition(raw_camera_viewer),
                parameters=[
                    {
                        "image_topic": "/track_preview_car/camera/image_raw",
                        "window_name": "Raw Track Car Camera",
                        "show_window": ParameterValue(camera_viewer_show_window, value_type=bool),
                        "display_width": 760,
                        "display_height": 428,
                        "window_x": 840,
                        "window_y": 40,
                        "sign_overlay_enabled": True,
                        "sign_detection_topic": "/lane_pid_experiment/camera_sign_detection",
                        "traffic_light_overlay_enabled": True,
                        "traffic_light_detection_topic": "/lane_pid_experiment/camera_traffic_light_state",
                        "overlay_topic": "/lane_pid_experiment/raw_camera_viewer_overlay",
                        "use_sim_time": True,
                    }
                ],
            ),
        ]
    )
