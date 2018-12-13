from celery import Celery
import configparser
from rpc import RpcClient
import pika
import celery
import os


"""Celery asynchronous task"""


# cache the rpc client for each task instantiation
class TanslationTask(celery.Task):
    conf = configparser.RawConfigParser()
    conf.read(os.environ("CONFIG_PROPERTIES"))
    _rpc_client = None

    @property
    def rpc_client(self):
        # channel declaration
        user = self.conf.get('config', 'user')
        password = self.conf.get('config', 'password')
        host = self.conf.get('config', 'host')
        port = self.conf.getint('config', 'port')
        durable = self.conf.getboolean('config', 'durable')
        exclusive = self.conf.getboolean('config', 'exclusive')
        auto_delete = self.conf.getboolean('config', 'auto_delete')
        callback_queue = self.conf.get('config', "callback_queue")
        publisher_queue = self.conf.get('config', "consumer_queue")

        credentials = pika.PlainCredentials(user, password)
        parameters = pika.ConnectionParameters(host=host,
                                               port=port,
                                               credentials=credentials)
        connection = pika.BlockingConnection(parameters)
        self._rpc_client = RpcClient(connection,
                                     callback_queue,
                                     publisher_queue,
                                     durable,
                                     exclusive,
                                     auto_delete)

        return self._rpc_client


# set up the broker
app = Celery("tasks",
             broker="amqp://{user:s}:{password:s}@{host:s}:{port:d}"
             .format(
                 user=os.environ('MQ_USER'),
                 password=os.environ('MQ_PASSWORD'),
                 host=os.environ('MQ_HOST'),
                 port=os.environ('MQ_PORT')),
             backend='amqp')


# make a asynchronous rpc request for translation
@app.task(name="tasks.translation", base=TanslationTask)
def translation(msg):

    return translation.rpc_client.call(msg)


if __name__ == '__main__':

    app.start()
