import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from skimage.io import imread
from skimage.transform import resize


# 1. Prepare data
input_dir = r'data\clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = resize(imread(img_path), (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)


data = np.asarray(data)
labels = np.asarray(labels)

# 2. Train / Test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)

# 3. Train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C':[1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)

# 4. Test performance
best_estimator = grid_search.best_estimator_

y_pred = best_estimator.predict(x_test)

score = accuracy_score(y_pred, y_test)

print(f'{score * 100}% of samples were correctly classified')

pickle.dump(best_estimator, open(r'model\classifier.pkl', 'wb'))