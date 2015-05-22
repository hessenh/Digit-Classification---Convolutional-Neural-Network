import tornado.ioloop
import tornado.web
import os
import sys
import main,random

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


class CNNSTARTHandler(tornado.web.RequestHandler):
    def post(self):

            numberOfImages = int(self.get_argument("numberOfImages"))
           
            # Get a list of "numberOfImages" numbers in range 59999(traing set)           
            listOfImages =  random.sample(range(59999),numberOfImages )
            # Add to dictionary
            l = {}
            for i in range(0,len(listOfImages)):
                l[i] = listOfImages[i]

            self.write(l)

class CNNIMAGEHandler(tornado.web.RequestHandler):
    def post(self):

            number = int(self.get_argument("number"))
            
            s = main.getNetworkImage(cnn,number)

            self.write(s)



application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/CNN", CNNHandler),
    (r"/CNNSTART",CNNSTARTHandler),
    (r"/CNNIMAGE",CNNIMAGEHandler),
    (r'/Webapp/(.*)', tornado.web.StaticFileHandler, {'path': settings["static_path"]})
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
