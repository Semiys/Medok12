# Импортируем необходимые библиотеки из задания
import os
import io
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from autoviz.AutoViz_Class import AutoViz_Class

# Проверка наличия файла
if not os.path.exists('heart.csv'):
    print("Ошибка: Файл heart.csv не найден в текущей директории")
    exit()

# 1. Загрузка набора данных
print("\nЗадание 1. Загрузка набора данных:")
with io.open('heart.csv', 'r', encoding='utf-8-sig') as file:
    data = file.read()

# Загружаем данные в DataFrame
df = pd.read_csv(io.StringIO(data))

# 2. Получение информации о наборе данных
print("\nЗадание 2. Получение информации о наборе данных:")
print("Информация о наборе данных:")
print(df.info())
print("\nПервые 5 строк данных:")
print(df.head())

# 3. Обработка пустых значений и дубликатов
print("\nЗадание 3. Обработка пустых значений и дубликатов:")
print("\nИсходные пустые значения:")
print(df.isnull().sum())

# Создаем второй набор с искусственными пропусками и дубликатами
df_modified = df.copy()
# Добавляем дубликаты
df_modified = pd.concat([df_modified, df_modified.iloc[:10]], ignore_index=True)
# Создаем пропуски
df_modified.loc[20:30, 'age'] = None
df_modified.loc[40:50, 'trestbps'] = None

print("\nИскусственные пропуски:")
print(df_modified.isnull().sum())
print("\nКоличество дубликатов:", df_modified.duplicated().sum())

# Обработка пропусков и дубликатов
df_modified = df_modified.dropna()
df_modified = df_modified.drop_duplicates()

# 4. Конструирование признаков
print("\nЗадание 4. Конструирование признаков:")
# Первый набор - только числовые признаки
df_numeric = df.select_dtypes(include=['float64', 'int64'])
print("\nНабор 1 - только числовые признаки:", df_numeric.columns.tolist())

# Второй набор - основные показатели здоровья
health_columns = ['age', 'trestbps', 'chol', 'thalach']
df_health = df[health_columns]
print("\nНабор 2 - основные показатели здоровья:", df_health.columns.tolist())

# Третий набор - важные диагностические признаки
diagnostic_columns = ['target', 'age', 'cp', 'thalach', 'exang']
df_diagnostic = df[diagnostic_columns]
print("\nНабор 3 - важные диагностические признаки:", diagnostic_columns)

# 5. Генерация нового набора и объединения
print("\nЗадание 5. Генерация нового набора и объединения:")
# Создаем уникальный идентификатор для корректного объединения
df['id'] = range(len(df))
df_new = df.sample(n=100).copy()
df_new['new_feature'] = range(len(df_new))

# Все виды объединений
merged_inner = pd.merge(df, df_new, on=['id', 'age'], how='inner')
merged_left = pd.merge(df, df_new, on=['id', 'age'], how='left')
merged_right = pd.merge(df, df_new, on=['id', 'age'], how='right')
merged_outer = pd.merge(df, df_new, on=['id', 'age'], how='outer')

print("\nРезультаты объединений:")
print("-" * 50)
print(f"Inner join: {len(merged_inner)} записей")
print(f"Left join: {len(merged_left)} записей")
print(f"Right join: {len(merged_right)} записей")
print(f"Outer join: {len(merged_outer)} записей")
print("-" * 50)

# 6. Группировка и агрегация
print("\nЗадание 6. Группировка и агрегация:")

g1 = df.groupby('target').mean()
print("\nГруппировка 1 (среднее по диагнозу):")
print(g1)

g2 = df.groupby('target').agg({
    'age': ['min', 'max', 'mean', 'std'],
    'trestbps': ['min', 'max', 'mean', 'std']
})
print("\nГруппировка 2 (агрегирующие функции):")
print(g2)

g3 = df.groupby('target').size()
print("\nГруппировка 3 (количество по диагнозу):")
print(g3)

g4 = df.groupby('target').median()
print("\nГруппировка 4 (медиана):")
print(g4)

g5 = df.groupby('target').quantile([0.25, 0.75])
print("\nГруппировка 5 (квартили):")
print(g5)

# 7. Новые признаки
print("\nЗадание 7. Создание новых признаков:")
print("\nОписание новых признаков:")
print("1. age_bp_ratio - отношение возраста к давлению")
print("2. heart_health_index - индекс здоровья сердца")
print("3. age_risk - возрастной фактор риска")

df_features = df.copy()
df_features['age_bp_ratio'] = df_features['age'] / df_features['trestbps']
df_features['heart_health_index'] = df_features['thalach'] / df_features['trestbps']
df_features['age_risk'] = df_features['age'] * (df_features['chol']/200)

print("\nПримеры значений новых признаков (первые 5 строк):")
print("-" * 50)
print(df_features[['age_bp_ratio', 'heart_health_index', 'age_risk']].head())
print("-" * 50)

# 8. Составной индекс
print("\nЗадание 8. Создание составного индекса:")
df_indexed = df_features.copy()
df_indexed.set_index(['target', 'age'], inplace=True)
print("\nПример данных с составным индексом (первые 5 строк):")
print("-" * 50)
print(df_indexed.head())
print("-" * 50)
print("\nСтруктура составного индекса:")
print(df_indexed.index.names)

# 9. Кодирование категориальных признаков
print("\nЗадание 9. Кодирование категориальных признаков:")
df_encoded = df.copy()
df_encoded['target_coded'] = df_encoded['target']  # Уже закодировано как 0/1
df_onehot = pd.get_dummies(df, columns=['target'])

print("\n1. Label Encoding (уже закодировано):")
print(df_encoded[['target', 'target_coded']].head())
print("\n2. One-Hot Encoding:")
print(df_onehot.filter(like='target').head())

# 10. Статистические данные
print("\nЗадание 10. Основные статистические показатели:")
print("\nОбщая информация:")
print(f"Количество записей: {len(df)}")
print(f"Количество признаков: {len(df.columns)}")
print("\nСтатистика по числовым признакам:")
print("-" * 50)
print(df.describe().round(3))
print("-" * 50)

# 11. Визуализация через pandas
print("\nЗадание 11. Визуализация через pandas:")

plt.figure(figsize=(10, 6))
df['age'].hist()
plt.title('Гистограмма распределения возраста')
plt.xlabel('Возраст')
plt.ylabel('Частота')
plt.savefig('histogram.png')
plt.close()

plt.figure(figsize=(10, 6))
df.plot(kind='scatter', x='age', y='trestbps')
plt.title('Диаграмма рассеивания возраст/давление')
plt.xlabel('Возраст')
plt.ylabel('Давление')
plt.savefig('scatter.png')
plt.close()

plt.figure(figsize=(10, 6))
df.boxplot(column='age', by='target')
plt.title('Диаграмма "ящик с усиками" для возраста по диагнозу')
plt.suptitle('')
plt.savefig('boxplot.png')
plt.close()

# 12. Pandas-Profiling отчет
print("\nЗадание 12. Создание отчета Pandas-Profiling")
df_report = df[[
    'age', 'sex', 'cp', 'trestbps', 'chol', 
    'fbs', 'restecg', 'thalach', 'exang', 
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]]

profile = ProfileReport(
    df_report, 
    title="Heart Disease Dataset Analysis",
    minimal=False,
    explorative=True
)
profile.to_file("heart_disease_report.html")

# 13. AutoViz графики
print("\nЗадание 13. Создание графиков AutoViz")
if not os.path.exists('autoviz_plots'):
    os.makedirs('autoviz_plots')

AV = AutoViz_Class()
dft = AV.AutoViz(
    filename="",
    sep=",",
    depVar="target",
    dfte=df,
    header=0,
    verbose=2,
    lowess=False,
    chart_format="png",
    save_plot_dir=os.path.abspath("autoviz_plots"),
    max_rows_analyzed=303,
    max_cols_analyzed=14
)

