import logging
from flask import Flask
from flask_cors import CORS
from flask_restful import Api

# instantiate the app
app = Flask(__name__)

CORS(app)
app.config.from_object(__name__)
api = Api(app)

# init logging
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)