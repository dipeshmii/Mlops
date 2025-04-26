import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('shelf_life_mlp.pkl', 'rb'))

# Input functions
def get_inputs():
    st.title("Shelf Life Prediction App (Smart Packaging)")
    storage_temp = st.number_input("Storage Temperature (°C)")
    storage_humid = st.number_input("Storage Humidity (%)")
    material_type = st.selectbox("Material Type", ["Glass", "HDPE", "PET", "PLA", "Paperboard"])
    product_state = st.selectbox("Product State", ["Semisolid", "Solid"])
    storage_condition = st.selectbox("Storage Condition", ["Chilled", "Refrigerated"])
    return {
        'Storage_Temperature': storage_temp,
        'Storage_Humidity': storage_humid,
        'Material_Type': material_type,
        'Product_State': product_state,
        'Storage_Condition': storage_condition
    }

# Prediction function
def predict(X_input):
    X_df = pd.DataFrame([X_input])
    prediction = model.predict(X_df)
    return prediction[0]

if __name__ == "__main__":
    X_input = get_inputs()
    if st.button("Predict Shelf Life"):
        result = predict(X_input)
        st.success(f"Predicted Shelf Life: {result:.2f} months")
