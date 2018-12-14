#!/bin/bash
BASE_DIR=/home/chenrihan/DipML
WORK_DIR=$BASE_DIR/serving
TASK=nmt_tasks
source ~/client/venv/bin/activate
export PYTHONPATH=$BASE_DIR/serving:$BASE_DIR/preprocess:$PYTHONPATH
export MQ_USER=nmt
export MQ_PASSWORD=dip_gpu
export MQ_HOST=127.0.0.1
export MQ_PORT=5672
export PROBLEM=translate_enzh_ai50k
export T2T_USR_DIR=$BASE_DIR/preprocess
export DATA_DIR=/home/chenrihan/t2t_data_2
export SERVERS="172.17.0.2:9000 172.17.0.2:9001"
export SERVABLE_NAMES="transformer transformer"
export TIMEOUT_SECS=1000
export MAX_RETRIES=4

celery -A $TASK worker --loglevel=INFO --concurrency=8
