from kombu import Exchange, Queue


exchange = Exchange("tasks_zhen")
exchange.durable = False
queue = Queue("tasks_zhen", exchange, routing_key="tasks_zhen")
queue.durable = False
queue.no_ack = True

CELERY_QUEUES = (
   queue,
)
CELERY_ROUTES = {
    'tasks_zhen.translation': {"queue": "tasks_zhen", "routing_key": "tasks_zhen"},
}
