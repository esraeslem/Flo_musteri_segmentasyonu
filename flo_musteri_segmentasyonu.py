##################################################################
# FLO - Gözetimsiz Öğrenme ile Müşteri Segmentasyonu
##################################################################

##################################################################
# İş Problemi
##################################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama
# stratejileri belirlemek istemektedir. Buna yönelik olarak müşterilerin
# davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre
# K-Means ve Hierarchical Clustering yöntemleriyle gruplar oluşturulacaktır.

##################################################################
# Veri Seti Hikayesi
##################################################################
# Veri seti FLO'dan son alışverişlerini 2020-2021 yıllarında OmniChannel
# (hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş
# alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.
#
# master_id                          : Eşsiz müşteri numarası
# order_channel                      : Alışveriş yapılan platforma ait kanal (Android, ios, Desktop, Mobile)
# last_order_channel                 : En son alışverişin yapıldığı kanal
# first_order_date                   : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date                    : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online             : Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline            : Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online        : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline       : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline  : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online   : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12        : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

##################################################################
# Gerekli Kütüphaneler
##################################################################
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # görselleri dosyaya kaydetmek için (arayüzsüz ortamlarda da çalışır)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import os
import warnings

warnings.simplefilter(action="ignore", category=Warning)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


##################################################################
# GÖREV 1: Veriyi Hazırlama
##################################################################

# Adım 1: flo_data_20k.csv verisini okutunuz.
df_ = pd.read_csv("data/flo_data_20k.csv")
df = df_.copy()

print("############### İlk 5 Gözlem ###############")
print(df.head())
print("############### Değişken Tipleri ###############")
print(df.dtypes)
print("############### Betimsel İstatistikler ###############")
print(df.describe().T)
print("############### Eksik Değer Kontrolü ###############")
print(df.isnull().sum())

# Tarih içeren değişkenleri datetime tipine çeviriyoruz.
date_columns = [col for col in df.columns if "date" in col]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Adım 2: Müşterileri segmentlerken kullanacağımız değişkenleri oluşturuyoruz.
# Not: Tenure (müşterinin yaşı) ve Recency (en son kaç gün önce alışveriş
# yaptığı) gibi yeni değişkenler türetiyoruz.

# Her müşterinin toplam alışveriş sayısı (online + offline)
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# Her müşterinin toplam harcaması (online + offline)
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Analiz tarihini veri setindeki en güncel alışveriş tarihinin
# birkaç gün sonrası olarak belirliyoruz.
analysis_date = df["last_order_date"].max() + pd.Timedelta(days=2)
print(f"\nAnaliz tarihi: {analysis_date.date()}")

# Recency: müşterinin en son kaç gün önce alışveriş yaptığı
df["recency"] = (analysis_date - df["last_order_date"]).dt.days

# Tenure: müşterinin ilk alışverişinden bu yana geçen gün sayısı (müşteri yaşı)
df["tenure"] = (analysis_date - df["first_order_date"]).dt.days

model_df = df[["order_num_total", "customer_value_total", "recency", "tenure"]]
print("\n############### Segmentasyonda Kullanılacak Değişkenler ###############")
print(model_df.describe().T)


##################################################################
# GÖREV 2: K-Means ile Müşteri Segmentasyonu
##################################################################

# Adım 1: Değişkenleri standartlaştırıyoruz.
# order_num_total ve customer_value_total değişkenleri oldukça çarpık
# (sağa yatık) dağıldığından önce log dönüşümü uygulayıp ardından
# StandardScaler ile standartlaştırıyoruz. Bu, uzaklık temelli (K-Means,
# Hierarchical Clustering) algoritmaların aykırı değerlerden daha az
# etkilenmesini sağlar.
print("\n############### Değişken Çarpıklıkları (log öncesi) ###############")
print(model_df.skew())

log_df = np.log1p(model_df)

sc = StandardScaler()
model_scaled = sc.fit_transform(log_df)
model_scaled_df = pd.DataFrame(model_scaled, columns=model_df.columns)

print("\n############### Standartlaştırılmış Değişkenler (ilk 5 satır) ###############")
print(model_scaled_df.head())

# Adım 2: Optimum küme sayısını belirliyoruz (Elbow / Dirsek Yöntemi).
sse = []
silhouette_scores_kmeans = []
K_range = range(2, 11)

for k in K_range:
    kmeans_k = KMeans(n_clusters=k, random_state=17, n_init=10)
    labels_k = kmeans_k.fit_predict(model_scaled_df)
    sse.append(kmeans_k.inertia_)
    silhouette_scores_kmeans.append(silhouette_score(model_scaled_df, labels_k))

print("\n############### K-Means: Küme Sayısına Göre SSE ve Silhouette Skoru ###############")
for k, s, sil in zip(K_range, sse, silhouette_scores_kmeans):
    print(f"k={k}: SSE={s:.1f}  Silhouette={sil:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(list(K_range), sse, "bo-")
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("SSE / Inertia")
plt.title("Optimum Küme Sayısı için Elbow (Dirsek) Yöntemi")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_elbow_yontemi.png", dpi=120)
plt.close()

# Elbow grafiği ve silhouette skorları incelendiğinde SSE düşüşünün
# k=5 civarında belirgin şekilde yavaşladığı görülmektedir.
# Bu nedenle K-Means için optimum küme sayısı k=5 olarak seçilmiştir.
optimum_k_kmeans = 5

# Adım 3: Modeli oluşturup müşterileri segmentliyoruz.
kmeans_final = KMeans(n_clusters=optimum_k_kmeans, random_state=17, n_init=10)
kmeans_final.fit(model_scaled_df)

df["kmeans_segment"] = kmeans_final.labels_
model_df["kmeans_segment"] = kmeans_final.labels_

print(f"\nK-Means (k={optimum_k_kmeans}) için Silhouette Skoru: "
      f"{silhouette_score(model_scaled_df, kmeans_final.labels_):.4f}")

# Adım 4: Her bir segmenti istatistiksel olarak inceliyoruz.
kmeans_summary = model_df.groupby("kmeans_segment").agg({
    "order_num_total": ["mean", "median", "count"],
    "customer_value_total": ["mean", "median"],
    "recency": ["mean", "median"],
    "tenure": ["mean", "median"],
})
print("\n############### K-Means Segment İstatistikleri ###############")
print(kmeans_summary)
kmeans_summary.to_csv(f"{OUTPUT_DIR}/kmeans_segment_istatistikleri.csv")

# Segment dağılımını görselleştiriyoruz.
plt.figure(figsize=(8, 5))
df["kmeans_segment"].value_counts().sort_index().plot(kind="bar", color="#f97316")
plt.xlabel("K-Means Segmenti")
plt.ylabel("Müşteri Sayısı")
plt.title("K-Means Segmentlerine Göre Müşteri Dağılımı")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_kmeans_segment_dagilimi.png", dpi=120)
plt.close()


##################################################################
# GÖREV 3: Hierarchical Clustering ile Müşteri Segmentasyonu
##################################################################

# Adım 1: Görev 2'de standartlaştırdığımız dataframe'i (model_scaled_df)
# kullanarak optimum küme sayısını dendrogram ve silhouette skoru ile
# belirliyoruz.
hc_complete = linkage(model_scaled_df, method="ward")

# 4 küme elde edecek şekilde kesim yüksekliğini, en büyük birleşme
# mesafeleri arasından (3. ve 4. en büyük mesafenin ortası) belirliyoruz.
merge_heights = np.sort(hc_complete[:, 2])[::-1]
cut_height = (merge_heights[2] + merge_heights[3]) / 2  # k=4 için kesim noktası

plt.figure(figsize=(14, 7))
dendrogram(
    hc_complete,
    truncate_mode="lastp",
    p=20,
    show_contracted=True,
    leaf_font_size=10,
    color_threshold=cut_height,
)
plt.title("Hiyerarşik Kümeleme Dendrogramı (Ward Bağlantısı)")
plt.xlabel("Gözlem Sayısı (yaprak düğümleri)")
plt.ylabel("Öklid Uzaklığı")
plt.axhline(y=cut_height, color="r", linestyle="--", label=f"Kesim Noktası (k=4, h={cut_height:.1f})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_dendrogram.png", dpi=120)
plt.close()

silhouette_scores_hc = []
K_range_hc = range(2, 9)
for k in K_range_hc:
    hc_k = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels_hc_k = hc_k.fit_predict(model_scaled_df)
    silhouette_scores_hc.append(silhouette_score(model_scaled_df, labels_hc_k))

print("\n############### Hierarchical Clustering: Küme Sayısına Göre Silhouette Skoru ###############")
for k, sil in zip(K_range_hc, silhouette_scores_hc):
    print(f"k={k}: Silhouette={sil:.4f}")

# Dendrogram ve silhouette skorları birlikte değerlendirildiğinde,
# en yorumlanabilir ve istikrarlı ayrım k=4 kümede elde edilmiştir.
optimum_k_hc = 4

# Adım 2: Modeli oluşturup müşterileri segmentliyoruz.
hc_final = AgglomerativeClustering(n_clusters=optimum_k_hc, linkage="ward")
hc_labels = hc_final.fit_predict(model_scaled_df)

df["hc_segment"] = hc_labels
model_df["hc_segment"] = hc_labels

print(f"\nHierarchical Clustering (k={optimum_k_hc}) için Silhouette Skoru: "
      f"{silhouette_score(model_scaled_df, hc_labels):.4f}")

# Adım 3: Her bir segmenti istatistiksel olarak inceliyoruz.
hc_summary = model_df.groupby("hc_segment").agg({
    "order_num_total": ["mean", "median", "count"],
    "customer_value_total": ["mean", "median"],
    "recency": ["mean", "median"],
    "tenure": ["mean", "median"],
})
print("\n############### Hierarchical Clustering Segment İstatistikleri ###############")
print(hc_summary)
hc_summary.to_csv(f"{OUTPUT_DIR}/hc_segment_istatistikleri.csv")

plt.figure(figsize=(8, 5))
df["hc_segment"].value_counts().sort_index().plot(kind="bar", color="#0ea5e9")
plt.xlabel("Hierarchical Clustering Segmenti")
plt.ylabel("Müşteri Sayısı")
plt.title("Hierarchical Clustering Segmentlerine Göre Müşteri Dağılımı")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_hc_segment_dagilimi.png", dpi=120)
plt.close()


##################################################################
# Sonuçların Karşılaştırılması ve Dışa Aktarılması
##################################################################
comparison = pd.crosstab(df["kmeans_segment"], df["hc_segment"])
print("\n############### K-Means ve Hierarchical Clustering Segmentlerinin Karşılaştırılması ###############")
print(comparison)
comparison.to_csv(f"{OUTPUT_DIR}/kmeans_vs_hc_karsilastirma.csv")

final_columns = [
    "master_id", "order_num_total", "customer_value_total",
    "recency", "tenure", "kmeans_segment", "hc_segment",
]
df[final_columns].to_csv(f"{OUTPUT_DIR}/musteri_segmentleri.csv", index=False)

print("\nAnaliz tamamlandı. Tüm grafikler ve özet tablolar 'outputs/' klasörüne kaydedildi.")
