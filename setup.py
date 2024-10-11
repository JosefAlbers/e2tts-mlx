from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f.readlines()]

setup(
    name='e2tts-mlx',
    url='https://github.com/JosefAlbers/e2tts-mlx',
    py_modules=['e2tts'],
    packages=find_packages(),
    version='0.0.4-alpha',
    readme="README.md",
    author_email="albersj66@gmail.com",
    description="Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS in MLX",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Josef Albers",
    license="Apache License 2.0",
    python_requires=">=3.12.3",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "e2tts = e2tts:fire_main",
        ],
    },
)
