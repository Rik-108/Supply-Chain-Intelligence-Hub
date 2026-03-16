import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import joblib

print("--- Starting HYBRID Demand Prediction Model Training (with Local Delivery feature) ---")

# --- 1. Data Loading ---
print("Loading and preparing data...")
try:
    orders = pd.read_csv('olist_orders_dataset.csv', encoding='utf-8-sig')
    order_items = pd.read_csv('olist_order_items_dataset.csv', encoding='utf-8-sig')
    products = pd.read_csv('olist_products_dataset.csv', encoding='utf-8-sig')
    customers = pd.read_csv('olist_customers_dataset.csv', encoding='utf-8-sig')
    translation = pd.read_csv('product_category_name_translation.csv', encoding='utf-8-sig')
    # NEW: Load sellers data to get seller location
    sellers = pd.read_csv('olist_sellers_dataset.csv', encoding='utf-8-sig')
except FileNotFoundError:
    print("Error: Ensure all Olist CSV files are present in the directory.")
    exit()

# --- 2. Merging Data ---
print("Merging datasets...")
df = orders.merge(order_items, on='order_id')
df = df.merge(products, on='product_id')
df = df.merge(customers, on='customer_id')
df = df.merge(translation, on='product_category_name', how='left')
# NEW: Merge with sellers data
df = df.merge(sellers, on='seller_id')


# --- 3. Feature Engineering and Aggregation ---
print("Engineering features and aggregating data...")
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df.dropna(subset=['product_category_name_english', 'price', 'customer_state', 'seller_state'], inplace=True)

df['month'] = df['order_purchase_timestamp'].dt.month
df['week'] = df['order_purchase_timestamp'].dt.to_period('W')

# NEW: Engineer the 'is_local_delivery' feature
df['is_local_delivery'] = (df['customer_state'] == df['seller_state']).astype(int)

# UPDATED: Aggregate to include the new local delivery ratio and remove freight
weekly_demand = df.groupby(['week', 'product_category_name_english', 'customer_state', 'month']).agg(
    units_sold=('order_item_id', 'count'),
    avg_price=('price', 'mean'),
    # NEW: Calculate the ratio of local deliveries for this group
    local_delivery_ratio=('is_local_delivery', 'mean')
).reset_index()

# --- 4. HYBRID Model Training ---
print("Training Stacking Ensemble (Hybrid) model...")
# UPDATED: Change the features list to use the new feature
features = ['product_category_name_english', 'customer_state', 'month', 'avg_price', 'local_delivery_ratio']
target = 'units_sold'

X = weekly_demand[features]
y = weekly_demand[target]

categorical_features = ['product_category_name_english', 'customer_state', 'month']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

estimators = [
    ('xgb', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
]

stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge()
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking_model', stacking_regressor)
])

model_pipeline.fit(X, y)

# --- 5. Saving the Model ---
print("Hybrid model training complete. Saving to 'demand_model.pkl'...")
joblib.dump(model_pipeline, 'demand_model.pkl')
print("--- Model saved successfully. You can now run the Streamlit app. ---")

