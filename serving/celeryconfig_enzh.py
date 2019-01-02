from kombu import Exchange, Queue


CELERY_QUEUES = (
    Queue("tasks_enzh", Exchange("tasks_enzh"), routing_key="tasks_enzh"),
)

CELERY_ROUTES = {
    'tasks_enzh.translation': {"queue": "tasks_enzh", "routing_key": "tasks_enzh"},
}
