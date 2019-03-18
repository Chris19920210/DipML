from celery import Celery
import celery
from nmt_utils import ZhEnNmtClient
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
    user_dict = os.environ["USR_DICT"]
    index = np.random.randint(len(servable_names))
    server = servers[index]
    servable_name = servable_names[index]
    _nmt_clients = []
    num_servers = len(servable_names)
    for server, servable_name in zip(servers, servable_names):
        _nmt_clients.append(ZhEnNmtClient(server,
                                          servable_name,
                                          t2t_usr_dir,
                                          problem,
                                          data_dir,
                                          user_dict,
                                          int(timeout_secs)))

    @property
    def nmt_clients(self):

        return self._nmt_clients

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logging.error('{0!r} failed: {1!r}'.format(task_id, exc))


# set up the broker
app = Celery("tasks_zhen",
             broker="amqp://{user:s}:{password:s}@{host:s}:{port:s}"
             .format(
                 user=os.environ['MQ_USER'],
                 password=os.environ['MQ_PASSWORD'],
                 host=os.environ['MQ_HOST'],
                 port=os.environ['MQ_PORT']),
             backend='amqp',
             task_serializer='json',
             result_serializer='json',
             accept_content=['json'],
             result_persistent=False
             )
app.config_from_object("celeryconfig_zhen")


# make a asynchronous rpc request for translation
@app.task(name="tasks_zhen.translation", base=TanslationTask, bind=True, max_retries=int(os.environ['MAX_RETRIES']))
def translation(self, msg):
    try:
        source = json.loads(msg, strict=False)
        target = json.dumps(translation.nmt_clients[os.getpid() % self.num_servers]
                            .query(source),
                            ensure_ascii=False).replace("</", "<\\/")
        return target

    except Exception as e:
        self.retry(countdown=self.request.retries, exc=e)


if __name__ == '__main__':
    app.start()
