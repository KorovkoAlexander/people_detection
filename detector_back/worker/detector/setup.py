from setuptools import setup, find_packages

setup(
    name='detector',
    version='0.1',
    description='SSDS detector for pytorch',
    author='Alexander Korovko',
    author_email='a.korovko@rambler-co.ru',

    packages=find_packages(),
    package_data={
        'detector': [
            'common/cfgs/*.yaml',
            'common/cfgs/default/*.yaml',
            'inference/tensorrt/cfgs/*.yaml'
        ],
        'resources': '*',
    },

    install_requires=[
        'aio-pika==5.5.3 ',
        'Click==7.0',
        'PyYAML==5.4',
        'albumentations==0.1.10',
        'filterpy==1.4.5',
        'imgaug==0.2.6',
        'msgpack==0.6.1',
        'numba==0.43.0',
        'numpy==1.15.0',
        'opencv-python==4.0.0.21',
        'pandas==0.23.4',
        #'pycuda==2018.1.1',
        'requests==2.21.0',
        'scikit-learn==0.20.3',
        'tensorboard==1.12.2',
        'tensorboardX==1.6',
        # 'tensorrt==5.1.2.2',
        'torch==1.0.1',
        'torchvision==0.2.1',
        'tqdm==4.29.1',
    ],
    python_requires='>=3.6.0'
)
