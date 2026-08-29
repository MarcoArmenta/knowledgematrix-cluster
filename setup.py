from setuptools import setup, find_packages

setup(
    name="knowledgematrix",
    version="0.1.1",
    description="Compute the knowledge matrices",
    author="Marco Armenta and Samuel Leblanc",
    maintainer="Samuel Leblanc",
    url="https://github.com/samueleblanc/knowledgematrix",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15"
    ],
    extras_require={
        "viz": ["matplotlib>=3.5"],
        # Only needed to load pretrained LLM weights (e.g. Qwen3.5) through
        # models/qwen3_5.py::Qwen3_5.from_huggingface.
        "hf": ["transformers>=5.15"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)