import os

import asyncio
import numpy as np

from aio_pika import connect_robust
from aio_pika.patterns import RPC
from functools import partial
from time import time

from PIL import Image
from detector.common.utils.config_parse import get_config
from detector.inference.ssds import ObjectDetector


def load_model():
    model_cfg = get_config(os.environ["MODEL_NAME"])

    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "detector",
        'resources',
        os.environ["MODEL_CHECKPOINT"]
    )

    return ObjectDetector(
        model_cfg,
        checkpoint_path,
        device_id=int(os.environ["DEVICE_ID"])
    )


class Worker:
    def __init__(self):
        self.model = load_model()

    @staticmethod
    async def predict(*, model, image):
        image = Image.open(image)
        img = np.array(image)

        if img.shape[2] > 3:
            img = img[:, :, :3]

        t_start = time()
        labels, scores, boxes = model.predict(
            img,
            threshold=float(os.environ['MODEL_THRESH'])
        )
        total_time = time() - t_start
        predicts = {
            'labels': labels,
            'scores': scores,
            'boxes': boxes,
            'time': total_time
        }
        return predicts

    async def rabbit_connection(self):
        user = os.environ["RABBITMQ_USER"]
        password = os.environ['RABBITMQ_PASS']
        host = os.environ['RABBITMQ_HOST']
        vhost = os.environ["RABBITMQ_VHOST"]
        connection = await connect_robust(
            f"amqp://{user}:{password}@{host}/{vhost}"
        )
        return connection

    async def consume_data(self):
        connection = await self.rabbit_connection()
        channel = await connection.channel()
        rpc = await RPC.create(channel)

        await rpc.register('predict', partial(self.predict, model=self.model), auto_delete=True)
        return connection


if __name__ == "__main__":
    worker = Worker()
    loop = asyncio.get_event_loop()
    connection = loop.run_until_complete(worker.consume_data())

    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(connection.close())
        loop.shutdown_asyncgens()

