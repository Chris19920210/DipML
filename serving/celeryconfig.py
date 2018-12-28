from kombu import Exchange, Queue


CELERY_QUEUES = (
    Queue("default", Exchange("default"), routing_key="default"),
    Queue("tasks_enzh", Exchange("tasks_enzh"), routing_key="tasks_enzh"),
    Queue("tasks_zhen", Exchange("tasks_zhen"), routing_key="tasks_zhen")
)

CELERY_ROUTES = {
    'tasks.translation_enzh': {"queue": "tasks_enzh", "routing_key": "tasks_enzh"},
    'tasks.translation_zhen': {"queue": "tasks_zhen", "routing_key": "tasks_zhen"}
}
