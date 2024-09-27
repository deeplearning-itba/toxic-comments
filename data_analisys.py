# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from helper import get_memory_size, get_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from itertools import cycle
# %%
folder = 'data/'
# %%
train = pd.read_csv(folder+"train.csv")
test = pd.read_csv(folder+"test.csv")
test_labels = pd.read_csv(folder+"test_labels.csv")
submission = pd.read_csv(folder+"sample_submission.csv")
# %%
list_classes = train.columns[2:]
list_classes
# %%
y = train[list_classes].values
print(y.shape)
print(y[:10])
# %%
X_train, X_valid, Y_train, Y_valid = train_test_split(train['comment_text'], y, test_size = 0.1)

print(X_train.shape, X_valid.shape)
print(Y_train.shape, Y_valid.shape)
# %%
raw_text_train = X_train.apply(str.lower)
raw_text_valid = X_valid.apply(str.lower)
raw_text_test = test["comment_text"].apply(str.lower)
# %%
max_features = 100_000

tfidf_vectorizer = TfidfVectorizer(max_df=0.11, min_df=1,
                                   max_features=max_features,
                                   stop_words='english')

tfidf_matrix_train = tfidf_vectorizer.fit_transform(raw_text_train)
# %%
len(tfidf_vectorizer.vocabulary_)
# %%
tfidf_matrix_train.shape
# %%
tfidf_matrix_valid = tfidf_vectorizer.transform(raw_text_valid)
# %%
sparsity = 1 - (tfidf_matrix_train>0).sum()/(tfidf_matrix_train.shape[0]*tfidf_matrix_train.shape[1])
# %%
print(sparsity)
# %%
np.save('data_processed/X_train_sparse.npy', tfidf_matrix_train)
np.save('data_processed/y_train.npy', Y_train)

np.save('data_processed/X_valid_sparse.npy', tfidf_matrix_valid)
np.save('data_processed/y_valid.npy', Y_valid)
# %%
# %%
dense_matrix_train = tfidf_matrix_train.todense()
dense_matrix_valid = tfidf_matrix_valid.todense()
# %%
get_memory_size(dense_matrix_train)
# %%
# Modelo
# %%
dense_matrix_train.shape[1], max_features, tfidf_matrix_train.shape[1]
# %%
Y_train.shape[1], len(list_classes)
# %%
model = get_model(
    max_features, len(list_classes), lambd=1e-4
)
model.summary()
# %%
loss, acc = model.evaluate(dense_matrix_train, Y_train, epochs=1024)
loss, acc
# %%
model.layers[0].weights[0].shape
# %%
batch_size = 1024
epochs = 10
# %%
pred_valid = model.predict(dense_matrix_valid, verbose = 1)
pred_train = model.predict(dense_matrix_train, verbose = 1)
# %%
print(roc_auc_score(Y_train, pred_train, average='macro'))
print(roc_auc_score(Y_valid, pred_valid, average='macro'))
# %%
history = model.fit(
    dense_matrix_train,
    Y_train, 
    batch_size=batch_size,
    epochs=epochs, 
    verbose=1, 
    validation_data=(dense_matrix_valid, Y_valid), 
)
# %%
from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
# %%
pred_valid = model.predict(dense_matrix_valid, verbose = 1)
pred_train = model.predict(dense_matrix_train, verbose = 1)
# %%
print(roc_auc_score(Y_train, pred_train, average='macro'))
print(roc_auc_score(Y_valid, pred_valid, average='macro'))
loss, acc
# %%
# 1000 features
# 0.9531687611569314
# 0.9516790063867973
# 10_000 features
# 0.9765747875241644
# 0.9707906336467719