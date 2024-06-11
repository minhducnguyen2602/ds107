from data_loader import DataLoader
from paper_model import Model

dataset_name = "binary_origin"
dataloader = DataLoader( dataset_name=dataset_name, extract_features=True)
dataloader.preprocess()
model = Model('svm')

model.train(dataloader.X_train, dataloader.y_train)
y_pred = model.predict(dataloader.X_test)
model.evaluate(dataloader.y_test, y_pred)