import cv2
import os
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
class DataLoader:
    def __init__(self, dataset_name: str = "binary", extract_features: bool = False):
        self.dataset_name = dataset_name
        self.extract_features = extract_features
        self.X = []
        self.y = []
        self.class_list = []
        if dataset_name == "binary_origin":
            self.get_data('binary_origin')
        elif dataset_name == "binary_augmented":
            self.get_data('binary_augmented')
        elif dataset_name == "multiclass":
            self.get_data('multiclass')
        else:
            raise ValueError("Invalid dataset name")
    def get_data(self, dataset_name):
        if dataset_name == 'binary_origin':
            self.data_dir = 'datasets/SalmonScan/SalmonScan/Raw'
        elif dataset_name == 'binary_augmented':
            self.data_dir = 'datasets/SalmonScan/SalmonScan/Augmented'
        elif dataset_name == 'multiclass':
            self.data_dir = 'datasets/archive/Freshwater Fish Disease Aquaculture in south asia/Train'
        print(f"Loading {dataset_name} dataset...")
        for class_id, class_name in enumerate(tqdm(os.listdir(self.data_dir))):
            self.class_list.append(class_name)
            class_path = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if self.extract_features:
                    img = self.feature_extractor(img)
                self.X.append(img)
                self.y.append(class_id)
    
    def preprocess(self, test_ratio: float = 0.2):
        print(f"Preprocessing {self.dataset_name} dataset...")
        #shuffle
        zipped = list(zip(self.X, self.y))
        import random
        random.shuffle(zipped)
        self.X, self.y = zip(*zipped)

        #split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_ratio, stratify=self.y, random_state=42)


    def feature_extractor(self, img):
        pass

# dataset_name = "binary_origin"
# dataloader = DataLoader( dataset_name=dataset_name, extract_features=False)
# dataloader.preprocess()