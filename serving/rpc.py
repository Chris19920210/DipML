import pika
import time
import json
import logging
import uuid


class RpcServer(object):
    def __init__(self, conf):
        self.conf = conf
        self.consumer_queue = None
        self.publisher_queue = None

    def server(self, callback, *args, **kwargs):
        # consume configuration
        self.consumer_queue = self.conf.get('config', "consumer_queue")

        self.publisher_queue = self.conf.get('config', "publisher_queue")

        # channel declaration
        user = self.conf.get('config', 'user')
        password = self.conf.get('config', 'password')
        host = self.conf.get('config', 'host')
        port = self.conf.getint('config', 'port')

        credentials = pika.PlainCredentials(user, password)
        parameters = pika.ConnectionParameters(host=host, port=port, credentials=credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        # queue configuration
        durable = self.conf.getboolean('config', 'durable')
        exclusive = self.conf.getboolean('config', 'exclusive')
        auto_delete = self.conf.getboolean('config', 'autodelete')

        channel.queue_declare(self.consumer_queue, durable=durable, exclusive=exclusive, auto_delete=auto_delete)

        channel.queue_declare(self.publisher_queue, durable=durable, exclusive=exclusive, auto_delete=auto_delete)

        channel.basic_qos(prefetch_count=1)
        no_ack = self.conf.getboolean('config', 'no_ack')

        def on_server_request(ch, method, _, body):
            response = callback(json.loads(body), *args, **kwargs)

            if not no_ack:
                ch.basic_ack(delivery_tag=method.delivery_tag)

            ch.basic_publish(exchange='',
                             routing_key=self.publisher_queue,
                             properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
                             body=json.dumps(response))
            logging.info("%s::req => '%s' response => '%s'" % (self.consumer_queue, body, response))

        # consumer configuration
        channel.basic_consume(on_server_request, no_ack=no_ack, exclusive=exclusive, queue=self.consumer_queue)

        print("%s RPC '%s' initialized" % (time.strftime("[%d/%m/%Y-%H:%M:%S]", time.localtime(time.time())),
                                           self.consumer_queue))
        channel.start_consuming()


class RpcClient(object):
    def __init__(self, conf):
        self.conf = conf
        self.callback_queue = self.conf.get('config', "callback_queue")
        self.publisher_queue = self.conf.get('config', "consumer_queue")
        # channel declaration
        self.user = self.conf.get('config', 'user')
        self.password = self.conf.get('config', 'password')
        self.host = self.conf.get('config', 'host')
        self.port = self.conf.getint('config', 'port')
        self.credentials = pika.PlainCredentials(self.user, self.password)
        self.parameters = pika.ConnectionParameters(host=self.host,
                                                    port=self.port,
                                                    credentials=self.credentials)
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()
        self.durable = self.conf.getboolean('config', 'durable')
        self.exclusive = self.conf.getboolean('config', 'exclusive')
        self.auto_delete = self.conf.getboolean('config', 'autodelete')
        self.channel.queue_declare(self.publisher_queue,
                                   durable=self.durable,
                                   exclusive=self.exclusive,
                                   auto_delete=self.auto_delete)

        self.channel.queue_declare(self.callback_queue,
                                   durable=self.durable,
                                   exclusive=self.exclusive,
                                   auto_delete=self.auto_delete)

        self.channel.basic_consume(self.on_response, no_ack=True,
                                   queue=self.callback_queue)

        self.response = None
        self.corr_id = None

    def on_response(self, ch, _, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body
        else:
            ch.basic_publish(exchange='',
                             routing_key=self.callback_queue,
                             properties=pika.BasicProperties(content_type="application/json"),
                             body=body)

    def call(self, msg):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key=self.publisher_queue,
                                   properties=pika.BasicProperties(
                                       content_type="application/json",
                                       reply_to=self.callback_queue,
                                       correlation_id=self.corr_id
                                   ),
                                   body=msg)
        while self.response is None:
            self.connection.process_data_events()
        return self.response
