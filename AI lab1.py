import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#1 балансирование выборки

input("y/n оценка баланса классов")
df = pd.read_csv('creditcard.csv')

X = df.drop(columns =['Class'])
y = df['Class']

print(f"баланс классов:\n{y.value_counts()}")

input("y/n применение методов балансирования")

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify = y
)

print(f"\nдо баланса:\n{y_train.value_counts()}")

X_under, y_under = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
print(f"\nпосле rus:\n{y_under.value_counts()}")
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
print(f"\nпосле smote:\n{y_smote.value_counts()}")

input("y/n сравнение до/после балансирования")
table = {
    'до баланса': y_train.value_counts(),
    'RandomUnderSampler': y_under.value_counts(),
    'SMOTE': y_smote.value_counts(),
}

print(f"\n{pd.DataFrame(table)}")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

data = [
    ('до баланса', y_train),
    ('RandomUnderSampler', y_under),
    ('SMOTE', y_smote)
]

for ax, (name, y_data) in zip(axes, data):
    counts = y_data.value_counts().sort_index()

    classes = ax.bar(
        ['Класс 0', 'Класс 1'],
        [counts.get(0, 0), counts.get(1, 0)],
        color = ['blue', 'red']
    )

    ax.set_title(name)
    ax.bar_label(classes, fmt='%.0f', padding=3)
    ax.set_ylabel('Количество объектов')

plt.suptitle('Распределение классов')
plt.tight_layout()
plt.show(block=False)

input("y/n оценка точности")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

models = {
    'без баланса': (X_train, y_train),
    'RandomUnderSampler': (X_under, y_under),
    'SMOTE': (X_smote, y_smote)
}

balance = []
for method, (X_res, y_res) in models.items():
    model = LogisticRegression(
        max_iter = 5000,
        random_state = 42
    )
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    balance.append({
      'метод' : method,
      'precision' : precision_score(y_test, y_pred),
      'recall' : recall_score(y_test, y_pred),
      'f1' : f1_score(y_test, y_pred)
    })
    

#2 деление выборки на обучающую и тестовую

input("y/n деление выборки на обучающую и тестовую")

print(f"\nбаланс:\n{pd.DataFrame(balance)}")

print("обучающая выборка:")
print(y_train.value_counts())

print("\nтестовая выборка:")
print(y_test.value_counts())

'''

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify = y
)

'''

#3 нормализация признаков

input("y/n обучение на X_train")

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler()
}

#сравнение метрик модели до и после масштабирования

scale_results = []
model = LogisticRegression(
    max_iter = 5000,
    random_state = 42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

scale_results.append({
    'метод': 'без нормализации',
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
})

for name, scaler in scalers.items():
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter = 5000,
        random_state = 42
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    scale_results.append({
        'метод': name,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    })
    
print(f"\nнормализация:\n{pd.DataFrame(scale_results)}")

#4 преобразования меняющие распределение признака

from sklearn.preprocessing import PowerTransformer, QuantileTransformer

input("y/n преобразования меняющие распределение признака")

transformers = {
    'PowerTransformer': PowerTransformer(),
    'QuantileTransformer': QuantileTransformer(
        n_quantiles = 1000,
        output_distribution = 'normal',
        random_state = 42
    )
}

transform_results = []
model = LogisticRegression(
    max_iter = 5000,
    random_state = 42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

transform_results.append({
    'метод': 'без преобразования',
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
})

transformed_data = {}

for name, transformer in transformers.items():
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    transformed_data[name] = (X_train_transformed, X_test_transformed)

    model = LogisticRegression(
        max_iter = 5000,
        random_state = 42
    )
    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)

    transform_results.append({
        'метод': name,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    })

print(f"\nпреобразование:\n{pd.DataFrame(transform_results)}")

amount_index = X_train.columns.get_loc('Amount')

data = [
    ('Исходное распределение', X_train['Amount'].values),
    ('PowerTransformer',
     transformed_data['PowerTransformer'][0][:, amount_index]),
    ('QuantileTransformer',
     transformed_data['QuantileTransformer'][0][:, amount_index])
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, values) in zip(axes, data):
    ax.hist(values, bins=50, edgecolor='black')
    ax.set_title(name)
    ax.set_xlabel('Значение Amount')
    ax.set_ylabel('Количество объектов')

plt.suptitle('Распределение Amount до и после преобразований')
plt.tight_layout()
plt.show(block=False)

#5 кросс-валидация

from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline

input("y/n кросс-валидация")

cv_pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),
    ('sampler', RandomUnderSampler(random_state=42)),
    ('model', LogisticRegression(max_iter=5000, random_state=42))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    cv_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='f1_macro'
)

print("\nРезультаты кросс-валидации:")
print(f"F1-macro по фолдам: {cv_scores}")
print(f"Средний F1-macro: {cv_scores.mean():.4f}")
print(f"Стандартное отклонение: {cv_scores.std():.4f}")


#6 понижение размерности

from sklearn.decomposition import PCA

input("y/n понижение размерности PCA")

pca_scaler = StandardScaler()
X_train_pca_scaled = pca_scaler.fit_transform(X_train)
X_test_pca_scaled = pca_scaler.transform(X_test)

pca_full = PCA()
X_train_pca_full = pca_full.fit_transform(X_train_pca_scaled)
X_test_pca_full = pca_full.transform(X_test_pca_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.searchsorted(cumulative_variance, 0.95) + 1

print(f"\nКоличество исходных признаков: {X_train.shape[1]}")
print(f"Количество компонент для сохранения 95% дисперсии: {n_components_95}")
print(f"Сохранённая дисперсия: {cumulative_variance[n_components_95 - 1]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker='o'
)
plt.axhline(
    y=0.95,
    linestyle='--',
    label='95% дисперсии'
)
plt.axvline(
    x=n_components_95,
    linestyle='--',
    label=f'{n_components_95} компонент'
)
plt.xlabel('Количество главных компонент')
plt.ylabel('Накопленная explained variance ratio')
plt.title('Выбор числа компонент PCA')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

X_train_pca = X_train_pca_full[:, :n_components_95]
X_test_pca = X_test_pca_full[:, :n_components_95]

pca_results = []

model_original = LogisticRegression(
    max_iter=5000,
    random_state=42
)
model_original.fit(X_train_pca_scaled, y_train)
y_pred_original = model_original.predict(X_test_pca_scaled)

pca_results.append({
    'метод': 'Без PCA',
    'число признаков': X_train_pca_scaled.shape[1],
    'precision': precision_score(y_test, y_pred_original),
    'recall': recall_score(y_test, y_pred_original),
    'f1': f1_score(y_test, y_pred_original)
})

model_pca = LogisticRegression(
    max_iter=5000,
    random_state=42
)
model_pca.fit(X_train_pca, y_train)
y_pred_pca = model_pca.predict(X_test_pca)

pca_results.append({
    'метод': 'PCA (95% дисперсии)',
    'число признаков': X_train_pca.shape[1],
    'precision': precision_score(y_test, y_pred_pca),
    'recall': recall_score(y_test, y_pred_pca),
    'f1': f1_score(y_test, y_pred_pca)
})

print("\nСравнение моделей до и после PCA:")
print(pd.DataFrame(pca_results).to_string(index=False))

input("конец?")