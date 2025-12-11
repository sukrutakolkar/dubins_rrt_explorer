from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dubins_rrt_explorer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sukrut',
    maintainer_email='sukrut@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'teleop_node = dublins_rrt_explorer.teleop_node:main',
            'planner_node = dublins_rrt_explorer.planner_node:main',
            'perception_node = dublins_rrt_explorer.perception_node:main',
            'controller_node = dublins_rrt_explorer.controller_node:main',
            'explorer_node = dublins_rrt_explorer.explorer_node:main',
        ],
    },
)
