import numpy as np
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

review_data = pd.read_csv('train.csv')
print(review_data.head())
text = review_data.text

with nlp.disable_pipes():
	vectors = np.array([nlp(review.text).vector for idx, review in review_data.iterrows()])

from sklearn.model_selection import train_test_split

print(vectors.shape)
print(review_data.target.shape)

#X_train, X_test, y_train, y_test = train_test_split(vectors, review_data.label, test_size=0.1, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(vectors, review_data['target'], train_size=0.8, test_size=0.2, random_state=0)

from sklearn.svm import LinearSVC

# Set dual=False to speed up training, and it's not needed
svc = LinearSVC(random_state=1, dual=False, max_iter=10000)
svc.fit(X_train, y_train)

print(f'Model test accuracy: {svc.score(X_valid, y_valid)*100:.3f}%')