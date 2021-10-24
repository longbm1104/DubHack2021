import numpy as np
import pickle
import cv2
import glob
import os
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.svm import SVC

dim = 100

class ShelfLifeCocos:

    def __init__(self, classes) -> None:
      self.weight = None
      self.classes = classes

    def getYourFruits(self, fruits, data_type, print_n=False, k_fold=False):
        images = []
        labels = []
        val = ['Training', 'Test']
        if not k_fold:
            path = "input/fruits-360/" + data_type + "/"
            for i,f in enumerate(fruits):
                p = path + f
                j=0
                count = 1
                for image_path in glob.glob(os.path.join(p, "*.jpg")):
                    label = None
                    if count <= 122:
                      label = "0.27"
                    elif count >= 123 and count <= 244:
                      label = "0.51"
                    elif count >= 225 and count <= 347:
                      label = "0.78"
                    else:
                      label = "0.96"
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (dim, dim))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    images.append(image)
                    labels.append(label)
                    count += 1
                    j+=1
                if(print_n):
                    print("There are " , j , " " , data_type.upper(), " images of " , fruits[i].upper())
            images = np.array(images)
            labels = np.array(labels)
            return images, labels
        else:
            for v in val:
                path = "input/fruits-360/" + v + "/"
                for i,f in enumerate(fruits):
                    p = path + f
                    j=0
                    for image_path in glob.glob(os.path.join(p, "*.jpg")):
                        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                        image = cv2.resize(image, (dim, dim))
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        images.append(image)
                        labels.append(i)
                        j+=1
            images = np.array(images)
            labels = np.array(labels)
            return images, labels

    def getAllFruits(self):
        fruits = []
        for fruit_path in glob.glob("input/fruits-360/Training/*"):
            fruit = fruit_path.split("/")[-1]
            fruits.append(fruit)
        return fruits

    #Choose your Fruits
    def train(self):
        fruits = ['Cocos'] #Binary classification

        #Get Images and Labels
        X_t, y_train =  self.getYourFruits(fruits, 'Training', print_n=True, k_fold=False)
        X_test, y_test = self.getYourFruits(fruits, 'Test', print_n=True, k_fold=False)
        # print(X_t, y_train)

        #Get data for k-fold
        X,y = self.getYourFruits(fruits, '', print_n=True, k_fold=True)

        #Scale Data Images
        scaler = StandardScaler()
        X_train = scaler.fit_transform([i.flatten() for i in X_t])
        X_test = scaler.fit_transform([i.flatten() for i in X_test])
        X = scaler.fit_transform([i.flatten() for i in X])

        svm = SVC(gamma='auto', kernel='linear', probability=True)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        self.weight = svm

        #Evaluation
        # precision = metrics.accuracy_score(y_pred, y_test) * 100
        # print("Accuracy with SVM: {0:.2f}%".format(precision))
        # cm , _ = plot_confusion_matrix(y_test, y_pred,classes=y_train, normalize=True, title='Normalized confusion matrix')
        # plt.show()

        # calculate the FPR and TPR for all thresholds of the classification
        # probs = svm.predict_proba(X_test)
        # probs = probs[:, 1]
        # svm_fpr, svm_tpr, thresholds = metrics.roc_curve(y_test, probs)
        # svm_auc = metrics.roc_auc_score(y_test, probs)

    def formatImage(self, image):
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # print(image)
        image = cv2.resize(image, (dim, dim))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        scaler = StandardScaler()
        flat_image = np.array([image.flatten()])
        # scaled_image = scaler.fit_transform(flat_image)
        return flat_image

    def predict(self, input_img):
        formatImg = self.formatImage(input_img)
        percent = (self.weight).predict(formatImg)[0]
        return percent

# model = ImageRecognition(['Cocos', 'Watermelon'])
# model.train()
# print(model.predict("input/fruits-360/Training/Cocos/45_100.jpg"))

# with open("imgPred_svm_model", 'rb') as f:
#     pickle.dump(model, f)

# joblib.dump(model, "imgPreg_svm_model.pkl")
# pickle.dump(model, open('svm_model.pkl', 'wb'))

def main():
    image = cv2.imread("input/fruits-360/Training/Cocos/45_100.jpg", cv2.IMREAD_COLOR)
    # model = ShelfLifeCocos(['Cocos', 'Watermelon'])
    # model.train()
    # print(model.predict(image))
    # pickle.dump(model, open('coco_model_2.pkl', 'wb'))
    model = pickle.load(open('coco_model_2.pkl', 'rb'))
    print(model.predict(image))
    # count = 0
    # for name in glob.glob('input/fruits-360/Training/Cocos/*'):
    #     count += 1
    # print(count)
    # print(glob.glob("/input/fruits-360/Training/Watermelon/*.jpg"))

# model = joblib.load("imgPreg_svm_model.pkl")
# model.predict("input/fruits-360/Training/Cocos/45_100.jpg")
if __name__ == "__main__":
    main()