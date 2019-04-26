import pika
import argparse


parser = argparse.ArgumentParser(description='queue_remover')
parser.add_argument('--host', type=str, default=None,
                    help='host')
parser.add_argument('--port', type=int, default=None,
                    help='port')
parser.add_argument('--username', type=str, default=None,
                    help='username')
parser.add_argument('--password', type=str, default=None,
                    help='password')
parser.add_argument('--queue', type=str, default=None,
                    help='queue')
args = parser.parse_args()


if __name__ == '__main__':
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=args.host,
            port=args.port,
            credentials=pika.PlainCredentials(username=args.username, password=args.password)
        )
    )
    channel = connection.channel()
    channel.queue_delete(queue=args.queue)
    channel.exchange_delete(exchange=args.queue)
    connection.close()
