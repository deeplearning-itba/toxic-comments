# %%
from custom_data_generator import CustomDataGenerator
import numpy as np
from helper import get_model
from sklearn.metrics import roc_auc_score
# %%
X_train_sparse = np.load(
    'data_processed/X_train_sparse.npy', allow_pickle=True
).item()
y_train = np.load('data_processed/y_train.npy')

X_valid_sparse = np.load(
    'data_processed/X_valid_sparse.npy', allow_pickle=True
).item()
y_valid = np.load('data_processed/y_valid.npy')
# %%
X_train_sparse.shape, y_train.shape, X_valid_sparse.shape, y_valid.shape
# %%
# %%
custom_data_gen = CustomDataGenerator(
    X_train_sparse,
    y_train,
    1024,
    shuffle=True
)
# %%
for X, y in custom_data_gen:
    break
# %%
X.shape, y.shape
# %%
model = get_model(
    X_train_sparse.shape[1],
    y_train.shape[1],
    lambd=1e-5
)
# %%
model.summary()
# %%
custom_data_gen_train = CustomDataGenerator(
    X_train_sparse,
    y_train,
    1024,
    shuffle=True
)
custom_data_gen_val = CustomDataGenerator(
    X_valid_sparse,
    y_valid,
    1024,
    shuffle=True
)
# %%
history = model.fit(
    custom_data_gen_train,
    epochs=10,
    validation_data=custom_data_gen_val,
    # callbacks=[early_stoping]
)
# %%
from matplotlib import pyplot as plt
# %%
history.history
# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# %%
# %%
custom_data_gen_train = CustomDataGenerator(
    X_train_sparse,
    y_train,
    1024,
    shuffle=False
)
custom_data_gen_val = CustomDataGenerator(
    X_valid_sparse,
    y_valid,
    1024,
    shuffle=False
)
# %%
pred_train = model.predict(custom_data_gen_train, verbose = 1)
pred_valid = model.predict(custom_data_gen_val, verbose = 1)
# %%
total_train_data = len(pred_train)
total_val_data = len(pred_valid)
# %%
total_train_data/X_train_sparse.shape[0]
# %%
print(roc_auc_score(y_train[:total_train_data], pred_train, average='macro'))
print(roc_auc_score(y_valid[:total_val_data], pred_valid, average='macro'))
# %%
