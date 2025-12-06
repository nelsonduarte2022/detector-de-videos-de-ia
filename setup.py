from setuptools import setup, find_packages

setup(
    name="deepfake-shield",
    version="1.0.0",
    description="Sistema de detección de videos deepfake con Deep Learning",
    author="Tu Nombre",
    author_email="tu-email@ejemplo.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "fastapi>=0.104.0",
        "streamlit>=1.29.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepfake-train=training.train:main",
            "deepfake-predict=predict:main",
            "deepfake-api=deployment.api:main",
        ],
    },
)
