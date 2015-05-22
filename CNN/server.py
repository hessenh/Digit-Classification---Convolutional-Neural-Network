import tornado.ioloop
import tornado.web
import os
import sys
import main

#
# Setting up CNN
#
cnn = main.getNetwork()


settings = {'debug': True, 
            'static_path': os.path.join(os.path.dirname(__file__), 'Webapp')}

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

    def post(self,data):
        self.write("9")


class CNNHandler(tornado.web.RequestHandler):
	def post(self):

            numbers = main.runCNN(cnn,self.get_argument("number"))
            
            self.write(numbers)




application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/CNN", CNNHandler),
    (r'/Webapp/(.*)', tornado.web.StaticFileHandler, {'path': settings["static_path"]})
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
