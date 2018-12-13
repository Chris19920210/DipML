import argparse
import json
import logging
import traceback
from tornado import web, ioloop, gen
"""Tornado Web Application"""



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
    def get(self):
        print(self.request.body)
        self.write(self.request.body)
        self.finish()

    def post(self):
        print(self.request.body)
        self.write(self.request.body)
        self.finish()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    #                     datefmt='%a, %d %b %Y %H:%M:%S',
    #                     filename='myapp.log',
    #                     filemode='w')
    application = web.Application([(r"/translation_v2", AsyncAppNmtHandler)])
    application.listen(port=5000)
    ioloop.IOLoop.instance().start()
