from glob import glob

from setuptools import find_packages, setup

package_name = "eeg_bci_pipeline"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Elias Mueller",
    maintainer_email="elias.mueller@pm.me",
    description="Mock EEG publisher and baseline decoder for the BCI pipeline.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mock_eeg_publisher = eeg_bci_pipeline.mock_eeg_publisher:main",
            "baseline_intent_decoder = eeg_bci_pipeline.baseline_intent_decoder:main",
            "intent_marker_publisher = eeg_bci_pipeline.intent_marker_publisher:main",
        ],
    },
)
