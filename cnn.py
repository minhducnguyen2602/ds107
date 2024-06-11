import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
import random
class CustomCNN:
    def __init__(self, img_height, img_width, num_classes, num_conv_layers):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.num_conv_layers = num_conv_layers
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        input_shape = (self.img_height, self.img_width, 3)
        
        # Add convolutional layers
        for i in range(self.num_conv_layers):
            if i == 0:
                model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            else:
                model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_data(self, images, labels, validation_ratio=0.2):
        images = np.array([cv2.resize(img, (self.img_width, self.img_height)) for img in images])
        return images, labels

    def fit(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, model_save_path='./best_model.keras'):
            # Thiết lập seed để đảm bảo kết quả tái lập
            seed = 42
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            y_train = to_categorical(y_train, num_classes=self.num_classes)
            y_val = to_categorical(y_val, num_classes=self.num_classes)

            # Custom callback to calculate macro F1 score
            class MacroF1Callback(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    y_val_pred = np.argmax(self.model.predict(X_val), axis=1)
                    y_val_true = np.argmax(y_val, axis=1)
                    macro_f1 = f1_score(y_val_true, y_val_pred, average='macro')
                    logs['macro_f1'] = macro_f1
                    print(f'Epoch {epoch+1}: Macro F1 Score: {macro_f1:.4f}')

            checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, mode='min')
            macro_f1_callback = MacroF1Callback()
            
            self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[checkpoint, macro_f1_callback]
        )
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return np.argmax(y_pred, axis=1)
    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))

# Example usage:
# images = [cv2.imread(path) for path in list_of_image_paths]
# labels = list_of_labels

# cnn = CustomCNN(img_height=128, img_width=128, num_classes=10, num_conv_layers=3)
# X_train, X_val, y_train, y_val = cnn.preprocess_data(images, labels)
# cnn.fit(X_train, y_train, X_val, y_val, epochs=20, model_save_path='best_model.h5')

