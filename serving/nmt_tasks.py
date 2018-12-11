from celery import Celery
import argparse
import configparser
from rpc import RpcClient

parser = argparse.ArgumentParser(description='remover')
parser.add_argument('--user', type=str, default=None,
                    help='user_name')
parser.add_argument('--password', type=str, default=None,
                    help='password')
parser.add_argument('--host', type=str, default=None,
                    help='host')
parser.add_argument('--port', type=int, default=None,
                    help='port')
parser.add_argument('--basic-config', type=str, default='./config.properties',
                    help='Path to Basic Configuration for RabbitMQ')
args = parser.parse_args()


app = Celery("tasks",
             broker="amqp://{user:s}:{password:s}@{host:s}:{port:d}"
             .format(
                 user=args.user,
                 password=args.password,
                 host=args.host,
                 port=args.port))


@app.task(name='task.translation')
def translation(msg):
    global rpc_client
    return rpc_client.call(msg)


if __name__ == '__main__':
    conf = configparser.RawConfigParser()
    conf.read(args.basic_config)
    rpc_client = RpcClient(conf)
    app.start()
