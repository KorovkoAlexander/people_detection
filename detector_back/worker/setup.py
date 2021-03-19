from setuptools import setup, find_packages

setup(
    name='detector-worker',
    version='0.1',
    description='people_detection_worker',
    author='Alexander Korovko',
    author_email='a.korovko@rambler-co.ru',

    packages=find_packages(),
    package_data={
        '.': '*.yaml'
    },

    install_requires=[
        'aio-pika==5.5.3 ',
        'detector==0.1',
        'Click==7.0',
        'PyYAML==3.13',
        'Pillow==8.1.1'
    ],
    python_requires='>=3.6.0'
)
