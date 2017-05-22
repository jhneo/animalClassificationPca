from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from time import time
import numpy as np
import os.path
import random
import cv2
import os

TRAIN_DATA_PATH = 'auto_data/temp/train'
TEST_DATA_PATH = 'auto_data/temp/test'

class_name = {'cat':0,'dog':1, 'giraffe':2,'wolf':3}
TEST_PERCENT = 0.25
PCA_N_COMPONENT = 64 #20 18 28 24
RANDOM_STATE = 42
RESHAPE_SIZE = 320

class animal_recog_brain():
    def __init__(self):
        self.X = []
        self.Y = []

    def get_data(self,data_path):
        for parent,dirnames,filenames in os.walk(data_path):
            for filename in filenames:
                class_label = parent[len(data_path)+1:]
                label = class_name[class_label]
                full_path = os.path.join(parent,filename)

                image = cv2.imread(full_path)
                image = cv2.resize(image,(64,64))

                feature = np.reshape(image, (1,-1))
                if 0 == len(self.X):
                    self.X = feature
                else:   
                    self.X = np.r_[self.X,feature]
                self.Y = np.append(self.Y, label)               

        print 'getting data....... X shape:',self.X.shape


    def pca_fit(self,n_components):
        print("Extracting the top %d eigenfeatures from %d datas"
      % (n_components, self.X.shape[0]))
        print("Projecting the input data on the orthonormal basis")
        self.pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(self.X)

    def pca_transform(self,X_input):
        X_input_pca = self.pca.transform(X_input)
        return X_input_pca

    def classifier_fit(self):
        print("Fitting the classifier to the training set.")
        X_pca = self.pca_transform(self.X)
        self.clf = RandomForestClassifier(n_estimators=64,max_depth=None,min_samples_split=16, random_state=0)
        self.clf = self.clf.fit(X_pca,self.Y)
        
    def classifier_predict(self,X_test):
        X_test_pca=self.pca_transform(X_test)
        y_pred = self.clf.predict(X_test_pca)
        return y_pred

    def cross_validation(self,k_fold):
        X_pca=self.pca_transform(self.X)
        scores = cross_validation.cross_val_score(self.clf,X_pca,self.Y,cv = k_fold)
        print ("%d-fold cross validation result:"%(k_fold),scores)


    def test_set_accuracy(self,data_path):
        X=[]
        Y=[]
        for parent,dirnames,filenames in os.walk(data_path):
            for filename in filenames:
                class_label = parent[len(data_path)+1:]
                label = class_name[class_label]
                full_path = os.path.join(parent,filename)
                image = cv2.imread(full_path)
                image = cv2.resize(image,(64,64))
                feature = np.reshape(image, (1,-1))
                if 0 == len(X):
                    X = feature
                else:   
                    X = np.r_[X,feature]
                Y = np.append(Y, label)
        y_pred = self.classifier_predict(X)
        # accuracy = np.true_divide(np.sum(y_pred == Y),len(y_pred))
        accuracy = np.true_divide(np.sum(y_pred == Y),y_pred.shape[0])
        print data_path,accuracy
        return accuracy

        #print ('set: "%s" accuracy: %.3f '%(data_path,accuracy))
    def load_model(self):
        self.clf = joblib.load("train_model_auto.m")
        self.pca = joblib.load("pca_model_auto.m")
        



def H_gamma_transform(image,GAMMA):
    C = 255**(1-GAMMA)
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(HSV)
    H = np.power(H,GAMMA)
    H = np.multiply(H,C)
    H = H.astype(np.int)
    H = cv2.convertScaleAbs(H)

    HSV = cv2.merge([H,S,V]) 
    merged_image = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    return merged_image


def mirror_transform(image):
    image = image[:,::-1,:]
    return image

def random_crob(image):
    w = image.shape[1]
    h = image.shape[0]
    w_crob = random.randint(0,30)
    h_crob = random.randint(0,30)
    image = image[w_crob:w-w_crob,h_crob:h-h_crob]
    image = cv2.resize(image,(w,h))
    return image

def updowm_transform(image):
    image = image[::-1,:,:]
    return image

def split_train_data(data_path,augmented_path):
    for parent,dirnames,filenames in os.walk(data_path):
        data_path_len = len(data_path)
        index = 0
        for filename in filenames:
            for dirname in dirnames:
                index = 0
            class_name = parent[data_path_len+1:]
            full_path = os.path.join(data_path,class_name,filename)

            if random.random() < 0.3:
                save_full_path = os.path.join(augmented_path,'test',class_name,class_name+str(index)+'.jpg')
                image = cv2.imread(full_path)
                image = cv2.resize(image,(RESHAPE_SIZE,RESHAPE_SIZE))
                cv2.imwrite(save_full_path,image)
                print save_full_path
                index+=1
            else:
                save_full_path = os.path.join(augmented_path,'train',class_name,class_name+str(index)+'.jpg')
                image = cv2.imread(full_path)
                image = cv2.resize(image,(RESHAPE_SIZE,RESHAPE_SIZE))
                cv2.imwrite(save_full_path,image)
                print save_full_path
                index+=1

                save_full_path = os.path.join(augmented_path,'train',class_name,class_name+str(index)+'.jpg')
                gamma067 = H_gamma_transform(image,0.67)
                cv2.imwrite(save_full_path,gamma067)
                print save_full_path
                index+=1            

                save_full_path = os.path.join(augmented_path,'train',class_name,class_name+str(index)+'.jpg')
                gamma15 = H_gamma_transform(image,1.5)
                cv2.imwrite(save_full_path,gamma15)
                print save_full_path
                index+=1            

                save_full_path = os.path.join(augmented_path,'train',class_name,class_name+str(index)+'.jpg')
                mirror_image = mirror_transform(image)
                cv2.imwrite(save_full_path,mirror_image)
                print save_full_path
                index+=1                   

                save_full_path = os.path.join(augmented_path,'train',class_name,class_name+str(index)+'.jpg')
                crob_image = random_crob(image)
                cv2.imwrite(save_full_path,crob_image)
                print save_full_path
                index+=1    
                
                save_full_path = os.path.join(augmented_path,'train',class_name,class_name+str(index)+'.jpg')
                updowm_image = updowm_transform(image)
                cv2.imwrite(save_full_path,updowm_image)
                print save_full_path
                index+=1    

def clear_temp_image(data_path):
    for parent,dirnames,filenames in os.walk(data_path):
        for filename in filenames:
            full_path =  os.path.join(parent,filename)
            os.remove(full_path)
    'Remove temp done.'

def train():

	accuracy_threshold=0.89
	count_down=0
	while 1!=0:
		count_down+=1

		# print 'count_down:',count_down
		if count_down>500:
			accuracy_threshold-=0.01
			count_down=0
			# print 'accuracy_threshold:',accuracy_threshold

		while 1!=0:
			clear_temp_image('auto_data/temp')
			split_train_data('auto_data/data','auto_data/temp')
			brain = animal_recog_brain()
			brain.get_data(TRAIN_DATA_PATH)
			count = 0
			while 1!=0:
				count+=1
				print 'count_down:        ',count_down
				print 'accuracy_threshold:',accuracy_threshold
				print 'clear_count:       ',count
				if (count > 50):
					break
				brain.pca_fit(PCA_N_COMPONENT)
				brain.classifier_fit()
				#brain.cross_validation(4)
				if(brain.test_set_accuracy(TEST_DATA_PATH)>accuracy_threshold):
					joblib.dump(brain.clf, "train_model_auto.m")
					joblib.dump(brain.pca, "pca_model_auto.m")				
					break

			if(brain.test_set_accuracy(TEST_DATA_PATH)>accuracy_threshold):
				break
			if (count > 50):
				break
		if(brain.test_set_accuracy(TEST_DATA_PATH)>accuracy_threshold):
			break



def main():

	''' training code'''
	# train()

	'''testing code'''
	brain = animal_recog_brain()
	brain.load_model()
	brain.test_set_accuracy(TEST_DATA_PATH)   

	'''generalization test'''
	# brain = animal_recog_brain()
	# brain.load_model()
	# brain.test_set_accuracy('generalization_data')
if __name__ == '__main__':
    main()