import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import nltk
from nltk.corpus import stopwords
import re
import joblib
import string
import numpy as np

print("--- Training & Evaluating Product Desirability Model (with Optimal Price Feature) ---")

# --- Setup & Data Loading ---
try:
    stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')
try:
    orders = pd.read_csv('olist_orders_dataset.csv', encoding='utf-8-sig')
    reviews = pd.read_csv('olist_order_reviews_dataset.csv', encoding='utf-8-sig')
    order_items = pd.read_csv('olist_order_items_dataset.csv', encoding='utf-8-sig')
    products = pd.read_csv('olist_products_dataset.csv', encoding='utf-8-sig')
    customers = pd.read_csv('olist_customers_dataset.csv', encoding='utf-8-sig')
    translation = pd.read_csv('product_category_name_translation.csv', encoding='utf-8-sig')
except FileNotFoundError:
    print("Error: Ensure all Olist CSV files are present.")
    exit()

df = orders.merge(reviews, on='order_id').merge(order_items, on='order_id').merge(products, on='product_id').merge(customers, on='customer_id').merge(translation, on='product_category_name', how='left')
df['is_return_proxy'] = (df['review_score'] <= 2).astype(int)
for col in ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']:
    df[col] = pd.to_datetime(df[col], errors='coerce')
df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
df['is_late'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)

# --- NEW: Feature Engineering for Price Optimality ---
category_avg_prices = df.groupby('product_category_name_english')['price'].mean().to_dict()
joblib.dump(category_avg_prices, 'category_avg_prices.pkl')
print("Saved category average prices to 'category_avg_prices.pkl'")

df['avg_category_price'] = df['product_category_name_english'].map(category_avg_prices)
# Calculate a non-linear score. It's 1 at the average price and decays towards 0.
# The 'sharpness' parameter controls how quickly the score drops.
def calculate_optimality(price, avg_price, sharpness=0.5):
    if pd.isna(avg_price) or avg_price == 0:
        return 0.5 # Neutral score if no average is available
    deviation = np.log(price / avg_price) if price > 0 and avg_price > 0 else 0
    return np.exp(-sharpness * (deviation ** 2))

df['price_optimality_score'] = df.apply(lambda row: calculate_optimality(row['price'], row['avg_category_price']), axis=1)


# --- NLP & Other Features ---
df['review_comment_message'].fillna('', inplace=True)
stop_words = set(stopwords.words('portuguese'))
def clean_text_portuguese(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return ' '.join(word for word in text.split() if word and word not in stop_words)
df['cleaned_comment'] = df['review_comment_message'].apply(clean_text_portuguese)

df['price'] = df['price'].replace(0, 0.01)
df['freight_ratio'] = df['freight_value'] / df['price']
df['shipping_pain_score'] = df['freight_ratio'] * df['delivery_time_days']

# --- Model Definition ---
# UPDATED: Use the new 'price_optimality_score' and remove old price features
numerical_features = ['price_optimality_score', 'shipping_pain_score']
categorical_features = ['product_category_name_english', 'customer_state', 'is_late']
text_feature = 'cleaned_comment'
df.dropna(subset=numerical_features + ['product_category_name_english'], inplace=True)
X = df[numerical_features + categorical_features + [text_feature]]
y = df['is_return_proxy']

preprocessor = ColumnTransformer(transformers=[('num', 'passthrough', numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features), ('nlp', TfidfVectorizer(max_features=500), text_feature)])

scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss', random_state=42)
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb_classifier)])

# --- Evaluation & Saving ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("\nModel Performance Metrics (with Optimal Price Feature):")
print(classification_report(y_test, y_pred, target_names=['Desired', 'Undesired']))
print(f"ROC AUC Score: {roc_auc:.3f}\n")

model_pipeline.fit(X, y)
joblib.dump(model_pipeline, 'desirability_model.pkl')
print("--- Desirability model saved successfully. ---")

