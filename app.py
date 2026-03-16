import streamlit as st

# --- Page Configuration ---
# This sets the title and icon that appear in the browser tab, and sets the layout to wide.
st.set_page_config(
    page_title="Supply Chain Intelligence Hub",
    page_icon="🚚",
    layout="wide"
)

# --- Main Page Content ---
st.title("Supply Chain Intelligence Hub 🚚")

st.markdown("""
Welcome to the integrated analytics dashboard. This hub provides two powerful tools to enhance supply chain decision-making.

**Please select a tool from the sidebar navigation to begin your analysis.**

---

### Tools Available:

-   **Demand Forecast:** A seasonal predictor to forecast weekly product demand based on category, location, and pricing. This tool helps with inventory and resource planning.
-   **Desirability Predictor:** An AI model to predict customer satisfaction and product desirability based on order details. This tool is crucial for managing product quality and reverse logistics.

This application uses machine learning models trained on the Olist E-Commerce dataset to provide predictive insights. To start, simply click on one of the pages listed in the sidebar.
""")

# Optional: Add an image for a better visual appeal on the home page
st.image(
    "https://images.unsplash.com/photo-1579532537598-459ecdaf39cc?q=80&w=2070&auto=format&fit=crop",
    caption="Data-Driven Supply Chain Optimization"
)

