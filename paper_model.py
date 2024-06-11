from sklearn.metrics import accuracy_score, classification_report

class Model:
    def __init__(self, model_name: str):
        if model_name == 'svm':
            from sklearn.svm import SVC
            # Set class_weight='balanced' to handle imbalanced data
            # self.model = SVC(kernel='linear', random_state=42)
            self.model = SVC(kernel='linear', class_weight='balanced', random_state=42)
        elif model_name == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier()
        elif model_name == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier()
        else:
            raise ValueError("Invalid model name")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
