from celery import Celery
import celery
from utils_nmt import NmtClient
import numpy as np
import json
import logging
import os

"""Celery asynchronous task"""


# cache the rpc client for each task instantiation
class TanslationTask(celery.Task):
    servers = os.environ['SERVERS'].split(" ")
    servable_names = os.environ['SERVABLE_NAMES'].split(" ")
    problem = os.environ["PROBLEM"]
    data_dir = os.environ["DATA_DIR"]
    timeout_secs = os.environ["TIMEOUT_SECS"]
    t2t_usr_dir = os.environ["T2T_USR_DIR"]
    index = np.random.randint(len(servable_names))
    server = servers[index]
    servable_name = servable_names[index]
    _nmt_clients = []
    num_servers = len(servable_names)
    for server, servable_name in zip(servers, servable_names):
        _nmt_clients.append(NmtClient(server,
                                      servable_name,
                                      t2t_usr_dir,
                                      problem,
                                      data_dir,
                                      int(timeout_secs)))

    @property
    def nmt_clients(self):

        return self._nmt_clients


# set up the broker
app = Celery("tasks",
             broker="amqp://{user:s}:{password:s}@{host:s}:{port:s}"
             .format(
                 user=os.environ['MQ_USER'],
                 password=os.environ['MQ_PASSWORD'],
                 host=os.environ['MQ_HOST'],
                 port=os.environ['MQ_PORT']),
             backend='amqp',
             task_serializer='json',
             result_serializer='json',
             accept_content=['application/json']
             )


# make a asynchronous rpc request for translation
@app.task(name="tasks.translation", base=TanslationTask, bind=True)
def translation(self, msg):
    logging.info(os.getpid())
    return json.dumps(translation.nmt_clients[os.getpid() % self.num_servers].query(json.loads(msg, strict=False)),
                      ensure_ascii=False).replace("</", "<\\/")


if __name__ == '__main__':
    app.conf.update(
        CELERY_TASK_SERIALIZER='json',
        CELERY_ACCEPT_CONTENT=['json'],  # Ignore other content
        CELERY_RESULT_SERIALIZER='json',
        CELERY_TIMEZONE='Europe/Oslo',
        CELERY_ENABLE_UTC=True, )

    app.start()
