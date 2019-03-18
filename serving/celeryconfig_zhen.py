from kombu import Exchange, Queue


CELERY_QUEUES = (
    Queue("tasks_zhen", Exchange("tasks_zhen"), routing_key="tasks_zhen", durable=False, no_ack=True),
)

CELERY_ROUTES = {
    'tasks_zhen.translation': {"queue": "tasks_zhen", "routing_key": "tasks_zhen"},
}
