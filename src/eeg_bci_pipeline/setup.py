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
    description="EEG decoding pipeline: mock and model-backed decoders, GDF replay, RViz markers, and offline CSP+LDA / EEGNet training.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mock_eeg_publisher = eeg_bci_pipeline.mock_eeg_publisher:main",
            "gdf_replay_publisher = eeg_bci_pipeline.gdf_replay_publisher:main",
            "mock_intent_decoder = eeg_bci_pipeline.mock_intent_decoder:main",
            "model_intent_decoder = eeg_bci_pipeline.model_intent_decoder:main",
            "evaluate_hand_classifier = eeg_bci_pipeline.training.hand_classifier_cli:main",
            "intent_marker_publisher = eeg_bci_pipeline.intent_marker_publisher:main",
        ],
    },
)
