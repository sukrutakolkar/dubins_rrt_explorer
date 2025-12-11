import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch_ros.actions import SetRemap
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    
    my_pkg = get_package_share_directory('dublins_rrt_explorer')
    slam_pkg = get_package_share_directory('slam_toolbox')
    config_file = os.path.join(my_pkg, 'config', 'slam_params.yaml')

    slam_stack = GroupAction([
        SetRemap(src='/map', dst='/slam_map'),
        SetRemap(src='/map_metadata', dst='/slam_map_metadata'),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam_pkg, 'launch', 'online_async_launch.py')
            ),
            launch_arguments={
                'slam_params_file': config_file,
                'use_sim_time': 'True'
            }.items()
        )
    ])

    return LaunchDescription([
        slam_stack
    ])