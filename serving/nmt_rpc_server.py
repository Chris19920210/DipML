import argparse
import configparser
import multiprocessing as mp
from rpc import RpcServer
from nmt_utils import validate_flags, NmtClient
import logging

"""Model server. one process one gpu """

argparser = argparse.ArgumentParser(description='configuration setting')
argparser.add_argument('--basic-config', type=str, default='./config.properties',
                       help='Path to Basic Configuration for RabbitMQ')
argparser.add_argument('--processes', type=int, default=4,
                       help='Num of Processes')
argparser.add_argument("--problem", type=str, default=None,
                       help="problem name")
argparser.add_argument("--data_dir", type=str, default=None,
                       help="path to data dir")
argparser.add_argument("--timeout_secs", type=int, default=100,
                       help="timeout-secs")
argparser.add_argument("--t2t_usr_dir", type=str, default=None,
                       help="path to data t2t_usr_dir")
argparser.add_argument("--servers", type=str, nargs="+", required=True,
                       help="servers list")
argparser.add_argument("--servable_names", type=str, nargs="+", required=True,
                       help="servers list")

args = argparser.parse_args()


def rpc_process(server, servable_name, t2t_usr_dir, problem, data_dir, timeout_secs):
    nmt_client = NmtClient(
                 server,
                 servable_name,
                 t2t_usr_dir,
                 problem,
                 data_dir,
                 timeout_secs)
    validate_flags(server, servable_name)
    conf = configparser.RawConfigParser()
    conf.read(args.basic_config)
    rpc_server = RpcServer(conf)
    rpc_server.server(nmt_client.query)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='myapp.log',
                        filemode='w')

    assert len(args.servable_names) == len(args.servers)
    assert len(args.servers) >= args.processes

    workers = [mp.Process(target=rpc_process,
                          args=(args.servers[i],
                                args.servable_names[i],
                                args.t2t_usr_dir,
                                args.problem,
                                args.data_dir,
                                args.timeout_secs
                                )) for i in range(args.processes)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
