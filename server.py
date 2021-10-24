from flask import Flask, render_template, request, abort
import pickle
import numpy as np
from imageRecognition import ImageRecognition
from watermelonTraining import ShelfLifeWatermelon
from cocosTraining  import ShelfLifeCocos
from PIL import Image
import cv2
import json
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = pickle.load(open('svm_model_2.pkl', 'rb'))
cocoModel = pickle.load(open('coco_model_2.pkl', 'rb'))
wtmlModel = pickle.load(open('wtml_model_2.pkl', 'rb'))
classes = ['Cocos', 'Watermelon']

# @app.route("/pathFile/<string:pathFile>")
# def helloWorld(pathFile):
#   fullPath = "input/fruits-360/Training/Cocos/" + pathFile
#   return model.predict(fullPath)

# @app.route("/hello")
# def helloWorld2():
#   return "HelloWork"

@app.route("/imgFile", methods=['POST'])
@cross_origin()
def imgRequestHandler():
    result_dict = {'output': 'output_key'}
    if request.method == 'POST':
      f = request.files['imgF']
      img = Image.open(f)
      img = np.array(img)
      img = cv2.resize(img,(224,224))
      img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
      fruit = model.predict(img)
      shelf_life = None
      if fruit == "cocos":
        shelf_life = cocoModel.predict(img)
      else:
        shelf_life = wtmlModel.predict(img)

      retObj = {"fruit": fruit, "shelf_life": shelf_life}
      return json.dumps(retObj)
    else:
      return ""


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=1104)