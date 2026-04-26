from glob import glob

from setuptools import find_packages, setup

package_name = "ch_milestones"

setup(
    name=package_name,
    version="0.0.2",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yl",
    maintainer_email="yl@example.com",
    description="Milestone data collection policies and producers for AIC.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ch_milestone_data_producer = ch_milestones.nodes.data_producer:main",
            "ch_milestone_environment_resetter = ch_milestones.nodes.environment_resetter:main",
        ],
    },
)
