# -*- coding: utf-8 -*-
"""
Breast Cancer Classification using XGBoost
Based on the paper:
"Breast Cancer Classification using XGBoost" by Rahmanul Hoque et al.
World Journal of Advanced Research and Reviews, 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# تنظیمات نمایش
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. بارگذاری داده‌ها
def load_data():
    """
    بارگذاری مجموعه داده سرطان پستان ویسکانسین
    """
    try:
        # بارگذاری داده‌ها از UCI repository
        from ucimlrepo import fetch_ucirepo
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
        
        # استخراج ویژگی‌ها و برچسب‌ها
        X = breast_cancer_wisconsin_diagnostic.data.features
        y = breast_cancer_wisconsin_diagnostic.data.targets
        
        print("✅ داده‌ها با موفقیت بارگذاری شدند")
        print(f"📊 شکل داده‌ها: {X.shape}")
        print(f"🎯 تعداد کلاس‌ها: {y.nunique()[0]}")
        
        return X, y
    
    except Exception as e:
        print(f"❌ خطا در بارگذاری داده‌ها: {e}")
        print("📥 بارگذاری از فایل محلی...")
        
        # مسیر جایگزین برای بارگذاری داده‌ها
        # در اینجا می‌توانید مسیر فایل CSV خود را قرار دهید
        try:
            data = pd.read_csv('../Docs/data.csv')
            X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, errors='ignore')
            y = data['diagnosis']
            return X, y
        except:
            print("❌ داده‌ها یافت نشدند. لطفاً مجموعه داده را دانلود کنید.")
            return None, None

# 2. پیش‌پردازش داده‌ها
def preprocess_data(X, y):
    """
    پیش‌پردازش داده‌ها
    """
    # کدگذاری برچسب‌ها (M=1, B=0)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # تقسیم داده‌ها به آموزش و آزمون (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("\n📈 اطلاعات تقسیم داده‌ها:")
    print(f"   داده‌های آموزشی: {X_train.shape[0]} نمونه")
    print(f"   داده‌های آزمون: {X_test.shape[0]} نمونه")
    print(f"   درصد داده‌های آزمون: {(X_test.shape[0]/(X_train.shape[0]+X_test.shape[0]))*100:.1f}%")
    
    return X_train, X_test, y_train, y_test

# 3. تجسم داده‌ها
def visualize_data(X, y):
    """
    تجسم ویژگی‌های داده‌ها
    """
    # تبدیل به DataFrame برای تجسم بهتر
    df = X.copy()
    df['diagnosis'] = y
    
    # تعداد ویژگی‌ها برای تجسم
    n_features = min(10, X.shape[1])
    features_to_plot = X.columns[:n_features]
    
    # نمودار جعبه‌ای برای ویژگی‌های انتخابی
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features_to_plot):
        if idx < len(axes):
            sns.boxplot(x='diagnosis', y=feature, data=df, ax=axes[idx])
            axes[idx].set_title(f'Boxplot of {feature}')
            axes[idx].set_xlabel('')
            axes[idx].set_xticklabels(['Benign', 'Malignant'])
    
    plt.suptitle('Boxplots of Selected Features by Diagnosis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # نقشه همبستگی
    plt.figure(figsize=(12, 10))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # توزیع ویژگی‌های مهم
    important_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 
                         'area_mean', 'concavity_mean']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(important_features):
        if idx < len(axes) and feature in X.columns:
            sns.histplot(data=df, x=feature, hue='diagnosis', kde=True, 
                        ax=axes[idx], element='step', stat='density')
            axes[idx].set_title(f'Distribution of {feature}')
            axes[idx].legend(['Benign', 'Malignant'])
    
    plt.suptitle('Distribution of Important Features', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# 4. مدل‌سازی با XGBoost
def train_xgboost(X_train, X_test, y_train, y_test):
    """
    آموزش مدل XGBoost
    """
    print("\n🚀 آموزش مدل XGBoost...")
    
    # تعریف پارامترهای مدل (مطابق مقاله)
    params = {
        'objective': 'binary:logistic',
        'learning_rate': 0.3,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_estimators': 100
    }
    
    # ایجاد و آموزش مدل
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # پیش‌بینی
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, y_pred, y_pred_proba

# 5. ارزیابی مدل
def evaluate_model(y_test, y_pred, y_pred_proba):
    """
    ارزیابی عملکرد مدل
    """
    # محاسبه معیارهای ارزیابی
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ماتریس درهم‌ریختگی
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n📊 نتایج ارزیابی مدل:")
    print("=" * 40)
    print(f"✅ دقت (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 صحت (Precision): {precision:.4f} ({precision*100:.2f}%)")
    print(f"🔍 بازیابی (Recall): {recall:.4f} ({recall*100:.2f}%)")
    print(f"⚖️  امتیاز F1: {f1:.4f} ({f1*100:.2f}%)")
    
    # نمایش ماتریس درهم‌ریختگی
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# 6. تجسم اهمیت ویژگی‌ها
def plot_feature_importance(model, feature_names):
    """
    تجسم اهمیت ویژگی‌ها
    """
    # استخراج اهمیت ویژگی‌ها
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # نمایش 15 ویژگی مهم
    top_features = feature_importance_df.head(15)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_features['Importance'], align='center')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance Score')
    plt.title('Top 15 Most Important Features', fontsize=16)
    plt.gca().invert_yaxis()
    
    # اضافه کردن مقدار عددی به میله‌ها
    for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # نمایش جدول اهمیت ویژگی‌ها
    print("\n🏆 رتبه‌بندی ویژگی‌های مهم:")
    print("=" * 50)
    for idx, row in top_features.iterrows():
        print(f"{row['Feature']:30} → {row['Importance']:.4f}")
    
    return feature_importance_df

# 7. مقایسه با سایر مدل‌ها
def compare_with_other_models():
    """
    مقایسه نتایج با سایر مدل‌ها (مطابق جدول مقاله)
    """
    comparison_data = {
        'Reference': ['[38]', '[39]', '[40]', '[40]', '[41]', '[42]', '[42]', 'Proposed'],
        'Algorithm': ['SVM', 'RF, K-stars, NN', 'Logistic Regression', 'Naive Bayes', 
                     'Decision Tree', 'XGBoost', 'Random Forest', 'XGBoost (Our)'],
        'Accuracy': [83.3, 61.85, 94.4, 92.3, 94.4, 74, 75, 94.74],
        'Samples': [256, 244, 569, 569, 569, 275, 275, 569],
        'Features': [5, 139, 32, 32, 32, 12, 12, 32]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # نمودار مقایسه
    plt.figure(figsize=(14, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(df_comparison)))
    
    bars = plt.bar(range(len(df_comparison)), df_comparison['Accuracy'], color=colors)
    plt.xticks(range(len(df_comparison)), df_comparison['Algorithm'], rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of Different Algorithms (Accuracy)', fontsize=16)
    plt.ylim([0, 100])
    
    # اضافه کردن مقادیر به میله‌ها
    for bar, acc in zip(bars, df_comparison['Accuracy']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc}%', ha='center', fontsize=10)
    
    # هایلایت کردن مدل پیشنهادی
    bars[-1].set_edgecolor('red')
    bars[-1].set_linewidth(3)
    
    plt.tight_layout()
    plt.show()
    
    return df_comparison

# تابع اصلی
def main():
    """
    تابع اصلی اجرای برنامه
    """
    print("=" * 60)
    print("🧬 Breast Cancer Classification using XGBoost")
    print("📄 Based on: Rahmanul Hoque et al. (2024)")
    print("=" * 60)
    
    # 1. بارگذاری داده‌ها
    X, y = load_data()
    if X is None or y is None:
        return
    
    # 2. پیش‌پردازش
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # 3. تجسم داده‌ها
    print("\n📊 در حال تجسم داده‌ها...")
    visualize_data(X, y)
    
    # 4. آموزش مدل XGBoost
    model, y_pred, y_pred_proba = train_xgboost(X_train, X_test, y_train, y_test)
    
    # 5. ارزیابی مدل
    results = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # 6. نمایش اهمیت ویژگی‌ها
    feature_importance_df = plot_feature_importance(model, X.columns.tolist())
    
    # 7. مقایسه با سایر مدل‌ها
    print("\n📈 در حال مقایسه با سایر مدل‌ها...")
    comparison_df = compare_with_other_models()
    
    print("\n" + "=" * 60)
    print("✅ اجرای برنامه با موفقیت به پایان رسید!")
    print("🎯 نتیجه نهایی:")
    print(f"   دقت مدل XGBoost: {results['accuracy']*100:.2f}%")
    print(f"   بازیابی: {results['recall']*100:.2f}%")
    print(f"   صحت: {results['precision']*100:.2f}%")
    print("=" * 60)
    
    return {
        'model': model,
        'results': results,
        'feature_importance': feature_importance_df,
        'comparison': comparison_df
    }

# اجرای برنامه
if __name__ == "__main__":
    # نصب کتابخانه‌های مورد نیاز در صورت نیاز
    try:
        import xgboost
        import seaborn
        import matplotlib
    except ImportError as e:
        print(f"📦 نصب کتابخانه مورد نیاز: {e}")
        print("لطفاً دستور زیر را اجرا کنید:")
        print("pip install xgboost seaborn matplotlib scikit-learn ucimlrepo")
    
    # اجرای برنامه اصلی
    main()