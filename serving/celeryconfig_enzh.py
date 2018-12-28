from kombu import Exchange, Queue


CELERY_QUEUES = (
    Queue("default", Exchange("default"), routing_key="default"),
    Queue("tasks_enzh", Exchange("tasks_enzh"), routing_key="tasks_enzh")
)

CELERY_ROUTES = {
    'tasks_enzh.translation': {"queue": "tasks_enzh", "routing_key": "tasks_enzh"},
}
