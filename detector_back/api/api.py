import os
from aio_pika import connect_robust
from aio_pika.patterns import RPC

from io import BytesIO
from aiohttp import web
from aiohttp.web_exceptions import HTTPBadRequest

import aiohttp_cors


class Handler:
    async def rabbit_connection(self):
        user = os.environ["RABBITMQ_USER"]
        password = os.environ['RABBITMQ_PASS']
        host = os.environ['RABBITMQ_HOST']
        vhost = os.environ["RABBITMQ_VHOST"]
        connection = await connect_robust(
            f"amqp://{user}:{password}@{host}/{vhost}"
        )
        return connection

    async def get_predictions(self, data):
        connection = await self.rabbit_connection()
        channel = await connection.channel()
        rpc = await RPC.create(channel)

        preds = await rpc.proxy.predict(image=data)
        return preds

    async def predictor_handler(self, request):
        reader = await request.multipart()

        field = await reader.next()
        assert field.name == 'image'

        with BytesIO() as f:
            while True:
                chunk = await field.read_chunk()  # 8192 bytes by default.
                if not chunk:
                    break
                f.write(chunk)
            f.seek(0)
            preds = await self.get_predictions(f)

        if preds is None:
            raise HTTPBadRequest()
        return web.json_response(preds)


handler = Handler()

if __name__ == '__main__':
    app = web.Application()
    cors = aiohttp_cors.setup(app)

    cors.add(
        app.router.add_route("POST", "/predict", handler.predictor_handler), {
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                max_age=3600,
            )
        })

    # app.add_routes([
    #     web.post('/predict', handler.predictor_handler)
    # ])
    web.run_app(app, host='0.0.0.0', port=8079)
