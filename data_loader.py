import cv2
import os
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from skimage.color import rgb2gray
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from sklearn.utils import shuffle

def create_feature_vector(img):
    gray_img = rgb2gray(img)
    flat_img = gray_img.flatten()
    mean_color = np.mean(flat_img)
    std_dev_color = np.std(flat_img)
    variance_color = np.var(flat_img)
    kurtosis_color = kurtosis(flat_img)
    skewness_color = skew(flat_img)

    glcm = graycomatrix((gray_img * 255).astype(np.uint8), distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0] 
    px = glcm / float(glcm.sum())
    entropy = -np.sum(px * np.log2(px + np.finfo(float).eps))
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    feature_vector = np.array([
        mean_color, std_dev_color, variance_color, kurtosis_color, skewness_color, contrast, correlation, energy, entropy, homogeneity
    ])  
    return feature_vector

def image_preprocessing_paper(img_goc):
    img = cv2.resize(img_goc, (600, 250), interpolation=cv2.INTER_CUBIC)
    image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3)
    img = clahe.apply(image_bw) + 15
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixel_values = img_lab.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    # Apply KMeans clustering
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(pixel_values)
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)
    segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2BGR)
    return segmented_image_bgr


class DataLoader:
    def __init__(self, dataset_name: str = "binary", extract_features: bool = False):
        self.dataset_name = dataset_name
        self.extract_features = extract_features
        self.class_list = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        if dataset_name == "binary_origin":
            self.X_train, self.y_train = self.get_data('binary_origin', "train")
            self.X_test, self.y_test =self.get_data('binary_origin', "val")
        elif dataset_name == "binary_augmented":
            self.X_train, self.y_train =self.get_data('binary_augmented', "train")
            self.X_test, self.y_test =self.get_data('binary_augmented', "val")
        elif dataset_name == "multiclass":
            self.X_train, self.y_train =self.get_data('multiclass' , "train")
            self.X_test, self.y_test =self.get_data('multiclass' , "val")
        else:
            raise ValueError("Invalid dataset name")
    def get_data(self, dataset_name, split):
        X = []
        y = []
        if dataset_name == 'binary_origin':
            self.data_dir = 'dataset_split/Raw' + '/' + split
        elif dataset_name == 'binary_augmented':
            self.data_dir = 'dataset_split/Augmented' + '/' + split
        elif dataset_name == 'multiclass':
            self.data_dir = 'dataset_split/Multiclass' + '/' + split
        print(f"Loading {dataset_name} {split} dataset...")
        for class_id, class_name in enumerate(tqdm(os.listdir(self.data_dir))):
            self.class_list.append(class_name)
            class_path = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if self.extract_features:
                    img = self.feature_extractor(img)
                X.append(img)
                y.append(class_id)
        return X, y
    
    # def preprocess(self, test_ratio: float = 0.2):
    #     print(f"Preprocessing {self.dataset_name} dataset...")
    #     #shuffle
    #     # zipped = list(zip(self.X, self.y))
    #     X_shuffled, Y_shuffled = shuffle(self.X, self.y, random_state=42)
    #     #split
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_shuffled, Y_shuffled, test_size=test_ratio, stratify= Y_shuffled, random_state=42)


    def feature_extractor(self, img):
        segmented_image_bgr = image_preprocessing_paper(img)
        feature_vector = create_feature_vector(segmented_image_bgr)
        return feature_vector

# dataset_name = "binary_origin"
# dataloader = DataLoader( dataset_name=dataset_name, extract_features=False)
# dataloader.preprocess()
# print(dataloader.y_train)