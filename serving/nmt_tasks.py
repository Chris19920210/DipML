from celery import Celery
import celery
from nmt_utils import EnZhNmtClient, ZhEnNmtClient
import numpy as np
import json
import logging
import os
"""Celery asynchronous task"""


# cache the rpc client for each task instantiation
class EnZhTanslationTask(celery.Task):
    servers = os.environ['E2Z_SERVERS'].split(" ")
    servable_names = os.environ['E2Z_SERVABLE_NAMES'].split(" ")
    problem = os.environ["E2Z_PROBLEM"]
    data_dir = os.environ["E2Z_DATA_DIR"]
    timeout_secs = os.environ["TIMEOUT_SECS"]
    t2t_usr_dir = os.environ["E2Z_T2T_USR_DIR"]
    index = np.random.randint(len(servable_names))
    server = servers[index]
    servable_name = servable_names[index]
    _nmt_clients = []
    num_servers = len(servable_names)
    for server, servable_name in zip(servers, servable_names):
        _nmt_clients.append(EnZhNmtClient(server,
                                          servable_name,
                                          t2t_usr_dir,
                                          problem,
                                          data_dir,
                                          int(timeout_secs)))

    @property
    def nmt_clients(self):

        return self._nmt_clients

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logging.error('{0!r} failed: {1!r}'.format(task_id, exc))


class ZhEnTanslationTask(celery.Task):
    servers = os.environ['Z2E_SERVERS'].split(" ")
    servable_names = os.environ['Z2E_SERVABLE_NAMES'].split(" ")
    problem = os.environ["Z2E_PROBLEM"]
    data_dir = os.environ["Z2E_DATA_DIR"]
    timeout_secs = os.environ["TIMEOUT_SECS"]
    t2t_usr_dir = os.environ["Z2E_T2T_USR_DIR"]
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
             accept_content=['json']
             )

app.config_from_object("celeryconfig")


# make a asynchronous rpc request for translation
@app.task(name="tasks.translation_enzh", base=EnZhTanslationTask, bind=True, max_retries=int(os.environ['MAX_RETRIES']))
def translation_enzh(self, msg):
    try:
        logging.info("Server is {0}".format(self.servers[os.getpid() % self.num_servers]))
        logging.info("Source:{0}".format(msg))
        source = json.loads(msg, strict=False)
        target = json.dumps(translation_enzh.nmt_clients[os.getpid() % self.num_servers]
                            .query(source),
                            ensure_ascii=False).replace("</", "<\\/")
        logging.info("Target:{0}".format(target))
        return target

    except Exception as e:
        logging.warning("Probably Server {0} got broken down".format(self.servers[os.getpid() % self.num_servers]))
        self.retry(countdown=2 ** self.request.retries, exc=e)


@app.task(name="tasks.translation_zhen", base=ZhEnTanslationTask, bind=True, max_retries=int(os.environ['MAX_RETRIES']))
def translation_zhen(self, msg):
    try:
        logging.info("Server is {0}".format(self.servers[os.getpid() % self.num_servers]))
        logging.info("Source:{0}".format(msg))
        source = json.loads(msg, strict=False)
        target = json.dumps(translation_zhen.nmt_clients[os.getpid() % self.num_servers]
                            .query(source),
                            ensure_ascii=False).replace("</", "<\\/")
        logging.info("Target:{0}".format(target))
        return target

    except Exception as e:
        logging.warning("Probably Server {0} got broken down".format(self.servers[os.getpid() % self.num_servers]))
        self.retry(countdown=2 ** self.request.retries, exc=e)


if __name__ == '__main__':

    app.start()
