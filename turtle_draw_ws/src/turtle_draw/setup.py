import glob
import os
from setuptools import find_packages, setup

package_name = 'turtle_draw'

# Include image assets only when they are present so colcon build never
# fails on a clean checkout that lacks the binary dog.jpg file.
_img_files = glob.glob(os.path.join('images', '*.*'))
_data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]
if _img_files:
    _data_files.append(
        (os.path.join('share', package_name, 'images'), _img_files)
    )

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=_data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Felipe Caiafa',
    maintainer_email='felipecaiafa0704@gmail.com',
    description='Draw image contours with turtlesim using a pure-NumPy CV pipeline',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'turtle_controller = turtle_draw.turtle_controller:main',
        ],
    },
)