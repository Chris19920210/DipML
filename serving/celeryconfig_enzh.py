from kombu import Exchange, Queue


exchange = Exchange("tasks_enzh")
exchange.durable = False
queue = Queue("tasks_enzh", exchange, routing_key="tasks_enzh")
queue.durable = False
queue.no_ack = True

CELERY_QUEUES = (
   queue,
)

CELERY_ROUTES = {
    'tasks_enzh.translation': {"queue": "tasks_enzh", "routing_key": "tasks_enzh"},
}
