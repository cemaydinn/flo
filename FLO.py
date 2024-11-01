import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Veri setini okuma
df = pd.read_csv("flo_data_20k.csv")

# Tarih değişkenlerinin tipini datetime olarak dönüştürme
date_columns = [col for col in df.columns if "date" in col]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Bugünün tarihini belirleme (veri setindeki son tarih baz alınarak)
today_date = df['last_order_date'].max() + pd.to_timedelta(2, 'D')

# Recency ve Tenure değişkenlerini oluşturma
df['Recency'] = (today_date - df['last_order_date']).dt.days
df['Tenure'] = (today_date - df['first_order_date']).dt.days

# Frequency ve Monetary değişkenlerini oluşturma
df['Frequency'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['Monetary'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']

# Bu değişkenlerin dağılımına bir göz atalım.
df[['Recency', 'Tenure', 'Frequency', 'Monetary']].describe()

from sklearn.preprocessing import StandardScaler

# Segmentasyon için kullanacağımız değişkenleri seçme
X = df[['Recency', 'Tenure', 'Frequency', 'Monetary']]

# Standartlaştırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# elbow
from sklearn.cluster import KMeans

inertia = []
K = range(1, 15)
for k in K:
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(X_scaled)
  inertia.append(kmeans.inertia_)

# Grafiği çizme
plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Küme Sayısı')
plt.ylabel('Inertia')
plt.title('Elbow Yöntemi ile Optimum Küme Sayısını Belirleme')
plt.show()

#silhoutte skor
from sklearn.metrics import silhouette_score

silhouette_scores = []
K = range(2, 15)
for k in K:
  kmeans = KMeans(n_clusters=k, random_state=42)
  labels = kmeans.fit_predict(X_scaled)
  score = silhouette_score(X_scaled, labels)
  silhouette_scores.append(score)

# Grafiği çizme
plt.figure(figsize=(8,5))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Küme Sayısı')
plt.ylabel('Silhouette Skoru')
plt.title('Silhouette Analizi ile Optimum Küme Sayısını Belirleme')
plt.show()

#Adım 3: Modeli Oluşturma ve Müşterileri Segmentleme
#Küme sayısını 4 ve modeli oluşturma

# KMeans modelini oluşturma
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Segmentleri ekleyelim
df['KMeans_Segment'] = kmeans.labels_

# Segmentlerin ortalama değerleri
segment_summary = df.groupby('KMeans_Segment')[['Recency', 'Tenure', 'Frequency', 'Monetary']].mean()

print(segment_summary)

# Segmentlerin sayısal dağılımı
segment_counts = df['KMeans_Segment'].value_counts()

print(segment_counts)

#Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
#Adım 1: Değişkenleri Standartlaştırma
#Adım 2: Optimum Küme Sayısını Belirleme (Dendogram)


import scipy.cluster.hierarchy as sch

plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Müşteriler')
plt.ylabel('Uzaklık')
plt.show()

#Adım 3: Modeli Oluşturma ve Müşterileri Segmentleme

from sklearn.cluster import AgglomerativeClustering

# Model oluşturma
hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
df['HC_Segment'] = hc.fit_predict(X_scaled)

#Adım 4: Her Bir Segmenti İstatistiksel Olarak İnceleme

# Segmentlerin ortalama değerleri
hc_segment_summary = df.groupby('HC_Segment')[['Recency', 'Tenure', 'Frequency', 'Monetary']].mean()

print(hc_segment_summary)

# Segmentlerin sayısal dağılımı
hc_segment_counts = df['HC_Segment'].value_counts()

print(hc_segment_counts)

