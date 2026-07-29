import streamlit as st
import requests
import pandas as pd

# This points to your FastAPI server running in the Colab background
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Medicine Finder", layout="centered")

st.title("💊 Medicine Price & Alternative Finder")
st.write("Find fair prices and cheaper alternatives for your prescribed medicines.")

# The simple search bar requested in the prompt
medicine_name = st.text_input("Enter Medicine Name (e.g., Mycodryl Syrup)")

if st.button("Search"):
    if medicine_name:
        with st.spinner("Fetching data..."):
            # Call your FastAPI endpoints
            price_response = requests.get(f"{API_URL}/predict-price", params={"medicine_name": medicine_name})
            alt_response = requests.get(f"{API_URL}/alternatives", params={"medicine_name": medicine_name})
            
            # Display Predicted Price
            if price_response.status_code == 200:
                price = price_response.json().get('predicted_price')
                st.success(f"**Predicted Fair Price:** ₹{price}")
            elif price_response.status_code == 404:
                st.error("Medicine not found in the database. Please check the spelling.")
            else:
                st.error("An error occurred while fetching the price.")
                
            # Display Alternatives and Chart
            if alt_response.status_code == 200:
                alts = alt_response.json().get('alternatives', [])
                if alts:
                    st.subheader("Cheaper Alternatives")
                    df_alts = pd.DataFrame(alts)
                    
                    # Display the list of cheapest 3-5 alternatives
                    st.dataframe(df_alts[['brand_name', 'manufacturer', 'price_inr']], use_container_width=True)
                    
                    st.subheader("Price Comparison Across Manufacturers")
                    # The small chart comparing prices requested in the prompt
                    st.bar_chart(data=df_alts, x='manufacturer', y='price_inr')
                else:
                    st.info("No cheaper alternatives found with the exact same composition.")
    else:
        st.warning("Please enter a medicine name.")
