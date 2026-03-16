import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Desirability Predictor", layout="wide")


# --- Caching Functions for Efficiency ---
@st.cache_resource
def load_model_and_prices():
    """Loads the desirability model and the pre-calculated average category prices."""
    try:
        model = joblib.load('desirability_model.pkl')
        avg_prices = joblib.load('category_avg_prices.pkl')
        return model, avg_prices
    except FileNotFoundError:
        st.error(
            "Model or price file not found! Please run `python train_desirability_model.py` from the root directory first.")
        st.stop()


@st.cache_data
def load_data():
    """Loads and prepares the raw data for UI controls and the context chart."""
    try:
        # Load only the columns needed for this page to optimize memory
        orders = pd.read_csv('olist_orders_dataset.csv', encoding='utf-8-sig', usecols=['order_id', 'customer_id'])
        reviews = pd.read_csv('olist_order_reviews_dataset.csv', encoding='utf-8-sig',
                              usecols=['order_id', 'review_score'])
        order_items = pd.read_csv('olist_order_items_dataset.csv', encoding='utf-8-sig',
                                  usecols=['order_id', 'product_id'])
        products = pd.read_csv('olist_products_dataset.csv', encoding='utf-8-sig',
                               usecols=['product_id', 'product_category_name'])
        customers = pd.read_csv('olist_customers_dataset.csv', encoding='utf-8-sig',
                                usecols=['customer_id', 'customer_state'])
        translation = pd.read_csv('product_category_name_translation.csv', encoding='utf-8-sig')

        df = orders.merge(reviews, on='order_id')
        df = df.merge(order_items, on='order_id')
        df = df.merge(products, on='product_id')
        df = df.merge(customers, on='customer_id')
        df = df.merge(translation, on='product_category_name', how='left')

        df.dropna(subset=['product_category_name_english', 'customer_state'], inplace=True)
        return df
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure all Olist CSV files are in the root directory.")
        st.stop()


# --- Load Model and Data ---
model, category_avg_prices = load_model_and_prices()
df = load_data()

# --- User Interface ---
st.title("🛍️ Product Desirability Predictor")
st.markdown("Use the controls to simulate an order and predict customer satisfaction.")

product_categories = sorted(df['product_category_name_english'].unique())
customer_states = sorted(df['customer_state'].unique())

col1, col2 = st.columns([1, 1.5])


def format_category_name(name):
    """Helper function to make category names user-friendly for the dropdown."""
    return name.replace('_', ' ').title()


# --- Column 1: Input Controls ---
with col1:
    st.header("Input Parameters")

    selected_category = st.selectbox("Product Category", product_categories, format_func=format_category_name,
                                     index=product_categories.index('bed_bath_table'))
    selected_state = st.selectbox("Customer State", customer_states, index=customer_states.index('SP'))
    price = st.slider("Price (BRL)", 10, 500, 120, 10, help="The selling price of the item.")

    freight_ratio_input = st.slider(
        "Freight Ratio (Cost / Price)", 0.0, 1.0, 0.20, 0.01,
        help="Shipping cost as a percentage of item price. e.g., 0.2 means 20%."
    )

    delivery_time = st.slider("Delivery Time (Days)", 1, 60, 14)
    is_late = st.toggle("Delivery was late",
                        help="Check this box if the order arrived after the estimated delivery date.")

# --- Column 2: Prediction and Visualization ---
with col2:
    st.header("Prediction & Context")

    # --- Feature Calculation in the Background ---
    # Calculate 'price_optimality_score'
    avg_price_for_category = category_avg_prices.get(selected_category, price)


    def calculate_optimality(price, avg_price, sharpness=0.5):
        if pd.isna(avg_price) or avg_price == 0:
            return 0.5
        deviation = np.log(price / avg_price) if price > 0 and avg_price > 0 else 0
        return np.exp(-sharpness * (deviation ** 2))


    optimality_score_input = calculate_optimality(price, avg_price_for_category)

    # Calculate the 'shipping_pain_score' interaction feature
    shipping_pain_score_input = freight_ratio_input * delivery_time

    # --- Create Input DataFrame for the Model ---
    # This must exactly match the features the model was trained on
    input_df = pd.DataFrame({
        'price_optimality_score': [optimality_score_input],
        'shipping_pain_score': [shipping_pain_score_input],
        'product_category_name_english': [selected_category],
        'customer_state': [selected_state],
        'is_late': [1 if is_late else 0],
        'cleaned_comment': [""]  # Placeholder as the app doesn't take text input
    })

    # Get the prediction from the model
    prob_undesired = model.predict_proba(input_df)[0][1]
    desirability_score = (1 - prob_undesired) * 100

    # Display the desirability score
    st.metric(label="Desirability Score", value=f"{desirability_score:.1f}%")
    st.progress(int(desirability_score))

    st.markdown("---")

    # Filter data to create the historical context chart
    category_reviews = df[df['product_category_name_english'] == selected_category][
        'review_score'].value_counts().sort_index()

    # Create the Plotly bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=category_reviews.index,
        y=category_reviews.values,
        name='Review Scores',
        marker_color=['crimson', 'salmon', 'lightslategray', 'royalblue', 'green']
    ))

    fig.update_layout(
        title=f"Historical Review Score Distribution for '{format_category_name(selected_category)}'",
        xaxis_title='Review Score (1-5 Stars)',
        yaxis_title='Number of Reviews'
    )
    st.plotly_chart(fig, use_container_width=True)

