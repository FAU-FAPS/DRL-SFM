import os
from glob import glob
from setuptools import setup

def find_data_files(source, destination):
    data_files = []
    for dirpath, _, filenames in os.walk(source):
        if filenames:
            relative_dir = os.path.relpath(dirpath, source)
            dest_dir = os.path.join(destination, relative_dir)
            files = [os.path.join(dirpath, f) for f in filenames]
            data_files.append((dest_dir, files))
    return data_files

package_name = 'hunav_rl'
cur_directory_path = os.path.abspath(os.path.dirname(__file__))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    # Add the shared library as package data.
    package_data={
        package_name: ['lightsfm.cpython-310-*.so']
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name,'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name,'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name,'worlds/'), glob('./worlds/*')),
    ] + find_data_files('models', os.path.join('share', package_name, 'models'))
      + find_data_files('worlds', os.path.join('share', package_name, 'worlds'))
      + find_data_files('config', os.path.join('share', package_name, 'config'))
      + find_data_files('urdf', os.path.join('share', package_name, 'urdf'))
      + find_data_files('maps', os.path.join('share', package_name, 'maps'))
      + find_data_files('config', os.path.join('share', package_name, 'config'))
      + find_data_files('rviz', os.path.join('share', package_name, 'rviz'))
      + find_data_files('behavior_trees', os.path.join('share', package_name, 'behavior_trees')),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='TODO@TODO.TODO',
    description=('This package creates a simulation in Gazebo which includes a differential drive '
                 'robot with Lidar and a Hospital world'),
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
          'start_training = hunav_rl.start_training:main',
          'trained_agent = hunav_rl.trained_agent:main',
          'unpause = hunav_rl.unpause:main',
          'eval = hunav_rl.eval:main',
          'path_follower_drlvo = hunav_rl.path_follower_drlvo:main',
          'path_follower_drlvo_retrained = hunav_rl.path_follower_drlvo_retrained:main',
          'path_follower_drlsf = hunav_rl.path_follower_drlsf:main',
          'proxemics_reward = hunav_rl.proxemics_reward:main',
          'scan_merge = hunav_rl.scan_merge:main',
          'start_evaluation = hunav_rl.start_evaluation:main',
          'start_multiple_evaluations = hunav_rl.start_multiple_evaluations:main',
        ],
    },
)