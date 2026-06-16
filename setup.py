from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ujamaa-multi-modal",
    version="0.1.0",
    author="Brian Tushae Thomas",
    author_email="brian@anthosintelligence.com",
    description="Ujamaa Multi-Modal: Cooperative Gated Recurrent Attention Foundation Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnthosIntelligence/ujamaa-multi-modal",
    project_urls={
        "Company": "https://anthosintelligence.com",
        "Documentation": "https://docs.anthosintelligence.com/ujamaa",
        "Source": "https://github.com/AnthosIntelligence/ujamaa-multi-modal",
    },
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "wandb>=0.15.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "pillow>=10.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "ujamaa-train=training.train:main",
            "ujamaa-chat=examples.chat:main",
            "ujamaa-viz=examples.visualize_routing:main",
        ],
    },
)
