from glob import glob
import os
from setuptools import find_packages, setup


package_name = "my_robot_vision"


def package_tree(source_dir):
    entries = []
    for path in glob(f"{source_dir}/**/*", recursive=True):
        if os.path.isfile(path):
            dest = os.path.join("share", package_name, os.path.dirname(path))
            entries.append((dest, [path]))
    return entries


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/maps", glob("maps/*")),
        (f"share/{package_name}/assets", glob("assets/*")),
    ]
    + package_tree("models")
    + package_tree("worlds")
    + package_tree("config"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Student",
    maintainer_email="student@example.com",
    description="ROS2 virtual line-following demo with camera, controller, and robot simulator nodes.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "line_follower = my_robot_vision.line_follower:main",
            "virtual_camera = my_robot_vision.virtual_camera:main",
            "robot_sim = my_robot_vision.robot_sim:main",
            "gazebo_track_car = my_robot_vision.gazebo_track_car:main",
            "gazebo_camera_viewer = my_robot_vision.gazebo_camera_viewer:main",
        ],
    },
)
