# Bilişsel Performans Skoru Tahmini

**Google Yapay Zeka Akademisi — YZTA 2026 Datathon**

Uyku, yaşam tarzı ve demografik verilerden yola çıkarak bireylerin **bilişsel performans skorunu** (`bilissel_performans_skoru`) tahmin eden bir regresyon çözümü. Çözüm; kapsamlı özellik mühendisliği, K-Fold target encoding, üç farklı gradient boosting modeli, seed averaging ve stacking ensemble üzerine kuruludur.

---

## Veri Seti

| Dosya | Boyut | Açıklama |
|---|---|---|
| `train.csv` | 56.000 × 24 | Eğitim verisi (hedef değişken dahil) |
| `test_x.csv` | 24.000 × 23 | Test verisi |
| `sample_submission.csv` | — | Örnek gönderim formatı |

**Hedef değişken:** `bilissel_performans_skoru` (0–10 aralığında sürekli değer)

**Öne çıkan özellikler:** uyku metrikleri (REM %, derin uyku %, uykuya dalma süresi, gece uyanma sayısı), stres skoru, günlük adım sayısı, kafein/ekran süresi, BMI, yaş, meslek, ülke, kronotip, ruh sağlığı durumu vb.

---

## Çözüm Mimarisi

### 1. Veri Temizleme — Ülke Normalizasyonu
Ülke sütununda aynı ülkenin İngilizce ve Türkçe yazımları karışık halde bulunuyordu (ör. `Spain` / `Ispanya`). 21 ülkelik bir eşleme sözlüğü ile tüm değerler Türkçe karşılıklarına normalize edildi.

### 2. Özellik Mühendisliği (~50 yeni özellik)
- **Uyku kalitesi:** `kaliteli_uyku` (REM + derin uyku), `rem_derin_oran`, `uyku_bozuklugu`, `uyku_verimlilik`, `uyku_skoru`
- **Stres:** kare/log dönüşümleri, düşük/yüksek stres bayrakları
- **Fiziksel aktivite:** log adım sayısı, aktif/hareketsiz bayrakları
- **Kafein & ekran:** log dönüşümleri, birleşik `kafein_ekran` skoru
- **Demografi:** BMI kategorileri, yaş grupları, yaş kare/log
- **Çevresel:** optimal oda sıcaklığı bayrağı, sıcaklık sapması
- **Sosyal jet lag:** hafta sonu uyku farkı mutlak değeri ve eşik bayrağı
- **Çapraz etkileşimler:** `stres × uyku bozukluğu`, `adım / stres`, `yaş × stres`, `kafein × uykuya dalma`, `nabız × stres`, `mesai × stres` vb.

Sonuç: 24 sütun → **79 özellik**

### 3. K-Fold Target Encoding
7 kategorik sütun (`meslek`, `ulke`, `kronotip`, `ruh_sagligi_durumu`, `cinsiyet`, `mevsim`, `gun_tipi`) için **10-fold, smoothing=10** parametreli target encoding uygulandı. Fold-bazlı yaklaşım sayesinde veri sızıntısı (leakage) önlendi.

### 4. Modeller
Üç gradient boosting modeli, düşük öğrenme oranı (**0.005**) ve **early stopping (100)** ile eğitildi:

| Model | Ana Parametreler |
|---|---|
| **LightGBM** | 5000 iter, num_leaves=127, subsample=0.75 |
| **XGBoost** | 5000 iter, max_depth=6, subsample=0.75 |
| **CatBoost** | 5000 iter, depth=7 |

### 5. Doğrulama Stratejisi
- **10-Fold Cross Validation** (5 fold yerine 10 — daha kararlı OOF tahminleri)
- **Seed Averaging:** 3 farklı seed (42, 123, 456) ile tüm pipeline tekrarlanıp ortalaması alındı → varyans azaltma

### 6. Ensemble — Stacking & Blend
OOF tahminleri üzerine üç farklı birleştirme stratejisi denendi ve en iyi OOF RMSE'ye sahip olan otomatik seçildi:

| Yöntem | OOF RMSE |
|---|---|
| **Ridge Stacking** | **1.21715** |
| ElasticNet Stacking | 1.21738 |
| Ağırlıklı Blend (1/RMSE) | 1.21902 |

Nihai tahminler `[0, 10]` aralığına kırpılarak (`np.clip`) submission dosyası oluşturuldu.

---

## Sonuçlar

| Model | Seed-Avg OOF RMSE |
|---|---|
| LightGBM | 1.22430 |
| XGBoost | 1.22202 |
| CatBoost | 1.21735 |
| **Ridge Stack (final)** | **1.21715**  |

Tekil en iyi model CatBoost olurken, Ridge tabanlı stacking tüm modellerin üzerine ek kazanç sağladı.

---

## Çalıştırma

### Gereksinimler
```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost
```

### Kaggle Ortamında
1. Notebook'a `yzta-2026-datathon` veri setini ekleyin.
2. Hücreleri sırasıyla çalıştırın:
   1. Veri yükleme
   2. Ülke normalizasyonu
   3. Özellik mühendisliği
   4. Target encoding
   5. Preprocessing (eksik doldurma + label encoding)
   6. Model parametreleri
   7. Seed averaging + 10-fold eğitim (⏱️ en uzun adım)
   8. Stacking & en iyi model seçimi
   9. Submission oluşturma
3. Çıktı: `/kaggle/working/submission.csv`

> **Not:** 3 seed × 10 fold × 3 model = 90 model eğitimi yapıldığı için toplam süre uzundur. Hızlı deneme için `SEEDS = [42]` ve `N_FOLDS = 5` olarak düşürülebilir.

---

## Proje Yapısı

```
├── notebook.ipynb          # Tüm pipeline (bu kod)
├── submission.csv          # Nihai tahminler (24.000 satır)
└── README.md
```

---

## İyileştirme Fikirleri

- [ ] Optuna ile hiperparametre optimizasyonu
- [ ] Nöral ağ (TabNet / MLP) ile ensemble çeşitliliği
- [ ] Pseudo-labeling
- [ ] Feature selection (permutation importance ile gereksiz özelliklerin elenmesi)

---

*YZTA 2026 Datathon kapsamında geliştirilmiştir.*
