#!/usr/bin/env python
# coding: utf-8

# # Описание проекта

# Чтобы оптимизировать производственные расходы, металлургический комбинат ООО «Так закаляем сталь» решил уменьшить потребление электроэнергии на этапе обработки стали. Вам предстоит построить модель, которая предскажет температуру стали.
# 
# Мы имеем данные об электродах, посредствам которых нагревается сплав, о добавках в виде сыпучих материалов, проволоки и газа и о результатах измерения темперутары на разных этапах технологического процесса. При этом известно, что во время самого процесса технологам доступна только температура после первого замера, следовательно, для обучения мы можем использовать только первый замер. Целевым же признаком является результат последнего замера.

# # Оглавление

# 1. [Анализ данных](#step1)  
# 2. [Подготовка данных](#step2)  
# 3. [Обучение моделей](#step3) 
# 4. [Оценка моделей](#step4) 

# ## Анализ данных <a id="step1"></a>  

# Получим данные и ознакомимся с ними более подробно

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[2]:


data_arc = pd.read_csv('/datasets/final_steel/data_arc.csv')
data_bulk = pd.read_csv('/datasets/final_steel/data_bulk.csv')
data_bulk_time = pd.read_csv('/datasets/final_steel/data_bulk_time.csv')
data_gas = pd.read_csv('/datasets/final_steel/data_gas.csv')
data_temp = pd.read_csv('/datasets/final_steel/data_temp.csv')
data_wire = pd.read_csv('/datasets/final_steel/data_wire.csv')
data_wire_time = pd.read_csv('/datasets/final_steel/data_wire_time.csv')


# ### Замеры температуры

# Так как мы можем использовать только первый и последний замер температуры, выделим только их из data_temp.

# In[3]:


start_temp = data_temp.groupby('key')['Время замера'].min()
start_temp_data = data_temp.merge(start_temp, on='Время замера')
last_temp = data_temp.groupby('key')['Время замера'].max()
last_temp_data = data_temp.merge(last_temp, on='Время замера')


# In[4]:


all_temp_data=start_temp_data.merge(last_temp_data, on='key').rename(columns = {'Время замера_x': 'start_meas_time', 
                                                                                'Время замера_y' : 'last_meas_time', 
                                                                                'Температура_x' : 'start_temp',
                                                                               'Температура_y': 'last_temp'})


# In[5]:


all_temp_data.info()


# Посмотрим на значения на гистограмме. Распределение похоже на нормальное.

# In[6]:


all_temp_data['start_temp'].plot.hist(bins=30, range=(1000, 2000), color= 'aquamarine', alpha=0.7)
all_temp_data['last_temp'].plot.hist(bins=30, range=(1000, 2000), color = 'coral', alpha=0.7)
plt.legend()
plt.show()


# Посмотрим на выбросы в данных

# **Первый замер**

# In[7]:


plt.ylim(1000, 1800)
all_temp_data.boxplot('start_temp')
plt.show()


# Значения меньше 1520 и больше 1650 являются выбросами


# In[ ]:





# **Последний замер**

# In[8]:


plt.ylim(1500, 1800)
all_temp_data.boxplot('last_temp')
plt.show()


# Значения меньше 1570 и больше 1620 являются выбросами.

# ### Газ

# Посмотрим на данные о поданном в сплав объеме газа. На гистограмме видно несколько выбросов на значении около 50

# In[9]:


data_gas['Газ 1'].plot.hist(bins=30, range=(0, 80))
plt.show()


# In[10]:


plt.ylim(0, 100)
data_gas.boxplot('Газ 1')
plt.show()


# Значения больше 25 можно считать выбросами. Но глядя на гистограмму, я бы исключила только значения больше 35 или 40.


# ### Данные об электродах

# Рассмотрим данные об активной и реактивной мощности.

# In[11]:


data_arc['Активная мощность'].plot.hist(bins=30, range=(0, 10), color= 'aquamarine')
data_arc['Реактивная мощность'].plot.hist(bins=30, range=(0, 10), color= 'coral', alpha=0.7)
plt.show()


# **Активная мощность**

# In[12]:


plt.ylim(0, 4)
data_arc.boxplot('Активная мощность')
plt.show()


# **Реактивная мощность**

# In[13]:


plt.ylim(0, 4)
data_arc.boxplot('Реактивная мощность')
plt.show()


# По диаграммам размаха можно сказать, что для активной мощности выбросами являются значения выше 1.5, а для реактивной значения выше 1.2. Но если смотреть на гистограмму, видим относительно правный спад, к тому же разброс значений в обоих случаях всего до 3-4, так что я бы пренебрегла этими выбросами, они кажутся не ошибками в данных, а просто разнообразными случаями.


# ## Подготовка данных <a id="step2"></a>  

# Для удобства возмем в качестве features не каждую можность отдельного нагрева а сумму всех мощностей для партии и количество нагревов.
# Остальные данные соединим с нашими температурами, чтобы получить общий датафрейм. Также подсчитаем количество времени которое прошло с первого до последнего замера.
# 
# В данных о примесях заменим пустые значения на нули, ведь они это и означают. Нет данных - нет примесей.
# 
# В общем датафрейме удалим ненужные столбцы, которые не будем использовать и переименуем русские названия на английские. Просто для единообразия.

# In[14]:


data_arc_count = data_arc.groupby(by='key').count()
data_arc_sum = data_arc.groupby(by='key').sum()


# In[15]:


data_wire = data_wire.fillna(0)
data_bulk = data_bulk.fillna(0)


# <div style="background: #cceeaa; padding: 5px; border: 1px solid green; border-radius: 5px;">
#     <font color='green'> <b><u>КОММЕНТАРИЙ РЕВЬЮЕРА</u></b>
# </font>
# <font color='green'><br>ОК, принято)

# In[16]:


all_data= all_temp_data.merge(data_arc_sum, on='key')
all_data = all_data.merge(data_arc_count['Начало нагрева дугой'], on='key')
all_data = all_data.merge(data_gas, on='key')
all_data = all_data.merge(data_bulk, on='key')
all_data = all_data.merge(data_wire, on='key')
all_data['start_meas_time'] = pd.to_datetime(all_data['start_meas_time'], format='%Y-%m-%dT%H:%M:%S')
all_data['last_meas_time'] = pd.to_datetime(all_data['last_meas_time'], format='%Y-%m-%dT%H:%M:%S')
all_data['time'] = (all_data['last_meas_time'] - all_data['start_meas_time']) // pd.Timedelta('1s')
all_data = all_data.drop(['start_meas_time', 'last_meas_time'], axis=1).rename(columns = {'Активная мощность': 'active_power_sum', 
                                                                                'Реактивная мощность' : 'reactive_power_sum', 
                                                                                'Начало нагрева дугой' : 'number_heatings',
                                                                                         'Газ 1': 'gas'})
all_data.info()


# Удалим те строки, у которых наш целевой признак (результат последнего замера температуры) оказался пустой.

# In[17]:


all_data = all_data.drop(all_data.loc[all_data['last_temp'].isna(),:,].index)
all_data



# Разделим данные на целевой признак и features.

# In[18]:


target = all_data['last_temp']
features = all_data.drop(['last_temp'], axis=1)


# Посмотрим на корреляцию данных. Так как данных много возьмем только основные. Потому что все фичи не влезут на график.

# In[19]:


data_corr = features.pivot_table(index = 'key', values = ['start_temp', 'active_power_sum', 
                                                          'reactive_power_sum', 'number_heatings', 'gas', 'time'])#, 
                                                         # 'Bulk 1', 'Bulk 2', 'Bulk 3', 'Bulk 4', 'Bulk 5', 'Bulk 6', 
                                                         # 'Bulk 7', 'Bulk 8', 'Bulk 9', 'Bulk 10', 'Bulk 11', 
                                                         # 'Bulk 12', 'Bulk 13', 'Bulk 14', 'Bulk 15', 'Wire 1', 
                                                         # 'Wire 2', 'Wire 3', 'Wire 4', 'Wire 5', 'Wire 6', 
                                                         # 'Wire 7', 'Wire 9', 'Wire 8'])
data_corr.corr()


# In[20]:


sns.heatmap(data_corr.corr(), cmap= 'coolwarm')
plt.show()


# Далее, разделим наши данные на тестовую и обучающую выборки. И удалим key так как он точно не является полезным.

# In[21]:


features = features.drop(['key'], axis=1)


# In[22]:


features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)


# Масштабируем данные

# In[23]:


numeric = ['start_temp', 'active_power_sum', 'reactive_power_sum', 'number_heatings', 'gas']

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])


# In[24]:


features


# ## Обучение моделей <a id="step3"></a>  

# Перед нами стоит задача регрессии, поэтому обучим несколько регрессионных моделей и посмотрим какая из них покажет наилучший результат.

# **Линейная регрессия**

# In[25]:


LinearReg = LinearRegression()
LinearReg.fit(features_train, target_train)
predictions_LinearReg = pd.DataFrame(LinearReg.predict(features_valid), columns=['LinearReg_predict'])


# **CatBoost**

# In[26]:


cat = CatBoostRegressor(loss_function='MAE', iterations=1, learning_rate=1, depth=10)
cat.fit(features_train, target_train)
predictions_cat = pd.DataFrame(cat.predict(features_valid), columns=['cat_predict'])


# **Random Forest Regressor**

# In[27]:


RandomForest = RandomForestRegressor(max_depth=30, n_estimators = 60, random_state=12345)
RandomForest.fit(features_train, target_train)
predictions_RandomForest = pd.DataFrame(RandomForest.predict(features_valid), columns=['RandomForest_predict'])


# **Decision Tree Regressor**

# In[28]:


DecisionTree = DecisionTreeRegressor(max_depth=6)
DecisionTree.fit(features_train, target_train)
predictions_DecisionTree = pd.DataFrame(DecisionTree.predict(features_valid), columns=['DecisionTree_predict'])


# **LightGBM**

# In[29]:


lgb_train = lgb.Dataset(features_train, target_train)
lgb_eval = lgb.Dataset(features_valid, target_valid, reference=lgb_train)

params = {'metric': {'mae'}}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=150,
                valid_sets=lgb_eval,
                early_stopping_rounds=30)
lgb_predictions = pd.DataFrame(gbm.predict(features_valid, num_iteration=gbm.best_iteration))


# ## Оценка моделей <a id="step4"></a>  

# In[30]:


print('MAE Linear Regression:', mean_absolute_error(target_valid, predictions_LinearReg))


# In[31]:


print('MAE CatBoostRegressor:', mean_absolute_error(target_valid, predictions_cat))


# In[32]:


print('MAE Random Forest Regressor:', mean_absolute_error(target_valid, predictions_RandomForest))


# In[33]:


print('MAE Decision Tree Regressor:', mean_absolute_error(target_valid, predictions_DecisionTree))


# In[34]:


print('MAE LightGBM:', mean_absolute_error(target_valid, lgb_predictions))


# Лучший результат показала модель **LightGBM**. Посмотрим на важность признаков для этой модели

# In[35]:


lgb.plot_importance(gbm)
plt.show()


# Видим, что 3 самых важных фактора это начачльная температура, суммарная активная мощность и время, которое прошло с 1 нагрева до последнего.

# Сравним показатели нашей моделей с Dummy моделью

# In[36]:


dummy_reg = DummyRegressor(strategy="mean")

dummy_reg.fit(features_train, target_train)
predictions_Dummy = pd.DataFrame(dummy_reg.predict(features_valid), columns=['dummy_predict'])
print('MAE Dummy:', mean_absolute_error(target_valid, predictions_Dummy))


# LightGBM очвидно лучше чем среднее значение.

# Для успокоения души посмотрим как выглядят наши предсказания на гистограмме и сравним их с настоящими замерами.

# In[37]:


target_valid.plot.hist(bins=100, range=(1500, 1700), color = 'coral', alpha=0.7)
lgb_predictions.plot.hist(bins=100, range=(1500, 1700), color = 'blue', alpha=0.7)


# Вывод: Выбираем модель LightGBM, которая показала наилучший результат метрики МАЕ = 5.74.
   




# # Отчет

# **Анализ данных**
# 
# Были проанализированы предоставленные признаки. 
# Для многих партий обнаружены пропуски в целевом признаке (результат последнего измерения температуры).
# Изучен разброс данных по гистограммам и диаграмме размаха для показателей:
# * Температура (первый и последний замер)
# * Объем газа
# * Активная мощность
# * Реактивная мощность
# 
# Во всех случаях распределение выглядит как нормальное. Выбросы есть, но больше похожи на длинные хвосты нормального распределения а не на ошибки в данных.
# 
# **Подготовка данных**
# * Выбросили серединные замеры температуры (со второй по предпоследнюю) так как использовать их мы не можем по услови задачи;
# * Активные и реактивные мощности всех этапов нагрева сложили в суммарную активную и суммарную реактивную можность;
# * Для каждой партии подсчитали количество нагревов;
# * Для каждой партии подсчитали временной интервал между первым и последним измерением температуры;
# * В данных о примесях (как сыпучих так и провологки) заменили Nan на нули, так как Nan означает отсутствие добавления примесей;
# * Изучили возможные корреляции в основных признаках, чтобы удалить признаки со слишком высокой корреляцией (таких не обнаружилось);
# * Масштабировали признаки.
# 
# Финальный датафрейм для обучения (features) по итогу содержит:
# * start_temp - результат первого замера температуры;
# * active_power_sum - суммарная активная мощность для партии;
# * reactive_power_sum - суммарная реактивная мощность для партии;
# * number_heatings - количество нагревов партии;
# * gas - объем газа;
# * time - разница во времени между первым и последним замером;
# * Bulk (1-15) - объем тех или иных сыпучих примесей в партии;
# * Wire (1-9) - объем проволочных примесей в партии.
# 
# Целевой признак (target) содержит результат последнего замера температуры партии
# 
# **Выбор модели и результаты обучения**
# 
# Для задачи регрессии обучили и сравнили результаты 5 моделей регрессии.
# 
# |Model         | MAE           |
# | ------------- |:-------------:|
# | Linear Regression      | 6.342183908965219 |
# | CatBoost Regressor      | 8.066895367884797 |
# | Random Forest Regressor | 5.950543167524298 |
# | Decision Tree Regressor | 7.12844910050853 |
# | LightGBM                | 5.746394525153782 |
# 
# Наилучший результат показала модель **LightGBM - 5.74**
# 
# **Выводы**
# По наилучшей модели посмотрели самые важные признаки. Топ 5 фичей:
# 1. start_temp
# 2. active_power_sum
# 3. time
# 4. Wire 1
# 5. reactive_power_sum
# 
# Все пункты плана были выполнены, дополнительно из существующих признаков были сгенерированы новые (количество нагревов, разница во времени между первым и последним замером). 
# 
# Основные трудности были связаны с пониманием процесса и, как следствие, с обработкой данных. Сложность с несколькими наблюдениями в рамках одной партии была решена обобщением признаков разных замеров (суммирование мощностей, подсчет количества итераций нагрева). Так победили)

# In[ ]:




