import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
import calendar

# --- Page Configuration ---
st.set_page_config(page_title="Demand Forecast", layout="wide")


# --- Caching Functions for Efficiency ---
@st.cache_resource
def load_model():
    """Loads the trained demand model from disk."""
    try:
        model = joblib.load('demand_model.pkl')
        return model
    except FileNotFoundError:
        st.error(
            "Model file 'demand_model.pkl' not found! Please run `python train_demand_model.py` from the root directory first.")
        st.stop()


@st.cache_data
def load_data():
    """Loads and prepares the raw data for UI controls and the context chart."""
    try:
        # Load all necessary datasets, including sellers for the new feature context
        orders = pd.read_csv('olist_orders_dataset.csv', encoding='utf-8-sig')
        order_items = pd.read_csv('olist_order_items_dataset.csv', encoding='utf-8-sig')
        products = pd.read_csv('olist_products_dataset.csv', encoding='utf-8-sig')
        customers = pd.read_csv('olist_customers_dataset.csv', encoding='utf-8-sig')
        translation = pd.read_csv('product_category_name_translation.csv', encoding='utf-8-sig')
        sellers = pd.read_csv('olist_sellers_dataset.csv', encoding='utf-8-sig')

        # Merge all datasets
        df = orders.merge(order_items, on='order_id')
        df = df.merge(products, on='product_id')
        df = df.merge(customers, on='customer_id')
        df = df.merge(translation, on='product_category_name', how='left')
        df = df.merge(sellers, on='seller_id')

        df.dropna(subset=['product_category_name_english', 'customer_state'], inplace=True)
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df['month'] = df['order_purchase_timestamp'].dt.month
        return df
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure all Olist CSV files are in the root directory.")
        st.stop()


# --- Load Model and Data ---
model = load_model()
df = load_data()

# --- User Interface ---
st.title("📈 Seasonal Product Demand Predictor")
st.markdown("Select product, customer, and time details to predict the weekly demand for a specific month.")

product_categories = sorted(df['product_category_name_english'].unique())
customer_states = sorted(df['customer_state'].unique())
month_names = list(calendar.month_name)[1:]
month_map = {name: i + 1 for i, name in enumerate(month_names)}

col1, col2 = st.columns([1, 1.5])

# --- Column 1: Input Controls ---
with col1:
    st.header("Input Parameters")


    def format_category_name(name):
        return name.replace('_', ' ').title()


    selected_category = st.selectbox("Product Category", product_categories, format_func=format_category_name,
                                     index=product_categories.index('bed_bath_table'))
    selected_state = st.selectbox("Customer State", customer_states, index=customer_states.index('SP'))
    selected_month_name = st.selectbox("Month of Year", month_names, index=pd.Timestamp.now().month - 1)
    selected_month_number = month_map[selected_month_name]
    price = st.slider("Average Price (BRL)", 10, 500, 120, 10)

    # UPDATED: Replaced the freight slider with the new local delivery ratio slider
    local_delivery_ratio = st.slider(
        "Local Delivery Ratio", 0.0, 1.0, 0.5, 0.05,
        help="The percentage of products sold from sellers within the same state as the customer. e.g., 0.5 means 50%."
    )

# --- Column 2: Prediction and Visualization ---
with col2:
    st.header("Prediction & Historical Context")

    # UPDATED: Pass the new 'local_delivery_ratio' to the model instead of freight
    input_df = pd.DataFrame({
        'product_category_name_english': [selected_category],
        'customer_state': [selected_state],
        'month': [selected_month_number],
        'avg_price': [price],
        'local_delivery_ratio': [local_delivery_ratio]
    })

    predicted_demand = model.predict(input_df)[0]

    st.metric(label=f"Predicted Weekly Units Sold in {selected_month_name}", value=f"{max(0, round(predicted_demand))}")
    st.markdown("---")

    hist_data = df[
        (df['product_category_name_english'] == selected_category) & (df['customer_state'] == selected_state)]

    if hist_data.empty:
        st.warning("No historical sales data found for this specific Category/State combination.")
    else:
        weekly_sales_hist = hist_data.set_index('order_purchase_timestamp').resample('W')[
            'order_item_id'].count().reset_index()
        weekly_sales_hist['month'] = weekly_sales_hist['order_purchase_timestamp'].dt.month
        weekly_sales_hist['color'] = weekly_sales_hist['month'].apply(
            lambda m: 'orange' if m == selected_month_number else 'royalblue')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weekly_sales_hist['order_purchase_timestamp'],
            y=weekly_sales_hist['order_item_id'],
            name='Weekly Units Sold',
            marker_color=weekly_sales_hist['color']
        ))

        fig.update_layout(
            title=f"Historical Sales for '{format_category_name(selected_category)}' in {selected_state} (<b>{selected_month_name} Highlighted</b>)",
            xaxis_title='Week',
            yaxis_title='Total Units Sold',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

