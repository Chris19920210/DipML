import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.httpclient
import tcelery
import nmt_tasks
from tornado.options import define, options
tcelery.setup_nonblocking_producer()


class BaseHandler(tornado.web.RequestHandler):
    def write_error(self, status_code, **kwargs):
        if status_code == 404:
            self.render("404.html")
        else:
            self.render("error.html")


class My404Handler(BaseHandler):
    def prepare(self):
        raise tornado.web.HTTPError(404)


class AsyncNmtHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        nmt_tasks.translation.apply_async(args=[self.request.body], callback=self.on_result)

    @tornado.web.asynchronous
    def post(self):
        nmt_tasks.translation.apply_async(args=[self.request.body], callback=self.on_result)

    def on_result(self, response):
        self.write(response.result)
        self.finish()
