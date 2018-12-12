import tornado.httpclient
import tcelery
import nmt_tasks
import argparse
import json
import logging
import traceback
from tornado import web, ioloop

parser = argparse.ArgumentParser(description='server')
parser.add_argument('--host', type=str, default=None,
                    help='host')
parser.add_argument('--port', type=int, default=None,
                    help='port')
args = parser.parse_args()

tcelery.setup_nonblocking_producer()


class MyAppException(web.HTTPError):

    pass


class MyAppBaseHandler(web.RequestHandler):

    def write_error(self, status_code, **kwargs):

        self.set_header('Content-Type', 'application/json')
        if self.settings.get("serve_traceback") and "exc_info" in kwargs:
            # in debug mode, try to send a traceback
            lines = []
            for line in traceback.format_exception(*kwargs["exc_info"]):
                lines.append(line)
            self.finish(json.dumps({
                'error': {
                    'code': status_code,
                    'message': self._reason,
                    'traceback': lines,
                }
            }))
        else:
            self.finish(json.dumps({
                'error': {
                    'code': status_code,
                    'message': self._reason,
                }
            }))


class AsyncAppNmtHandler(MyAppBaseHandler):
    @tornado.web.asynchronous
    def get(self):
        content_type = self.request.headers.get('Content-Type')
        if not (content_type and content_type.lower().startswith('application/json')):
            MyAppException(reason="Wrong data format, needs json", status_code=400)
        nmt_tasks.translation.apply_async(args=[self.request.body], callback=self.on_result)

    @tornado.web.asynchronous
    def post(self):
        content_type = self.request.headers.get('Content-Type')
        if not (content_type and content_type.lower().startswith('application/json')):
            MyAppException(reason="Wrong data format, needs json", status_code=400)
        nmt_tasks.translation.apply_async(args=[self.request.body], callback=self.on_result)

    def on_result(self, response):
        self.write(response.result)
        self.finish()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='myapp.log',
                        filemode='w')
    application = tornado.web.Application([r"/translation", AsyncAppNmtHandler])
    application.listen(port=args.port, address=args.host)
    ioloop.IOLoop.instance().start()
