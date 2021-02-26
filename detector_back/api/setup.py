from setuptools import setup, find_packages

setup(
    name='detector-api',
    version='0.1',
    description='people_detection_api',
    author='Alexander Korovko',
    author_email='a.korovko@rambler-co.ru',

    packages=find_packages(),

    install_requires=[
        'aio-pika==5.5.3 ',
        'aiohttp==3.7.4',
        'aiohttp-cors==0.7.0'
    ],
    python_requires='>=3.6.0'
)
