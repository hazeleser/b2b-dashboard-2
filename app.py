#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt


st.markdown("""
        <style>

        div[data-testid="stPlotlyChart"] {
            background-color: #fff6f1;
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            border: 1px solid #e2e8f0;
        }
        </style>
        """, unsafe_allow_html=True)

card_color = "#fff6f1"

st.balloons()

st.set_page_config(page_title="Sales Dashboard", layout="wide")

#tab eklendi
tab1, tab2, tab3, tab4= st.tabs(["General Look", "Descriptive Statistics", "ABC-XYZ Analysis", "Data Look UP"])

# Data Load
@st.cache_data
def load_data():
    df = pd.read_excel(r"/Users/edibenevagurbuz/Desktop/python/B2B_Transaction_Data.xlsx")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    # Sayısal kolonlar
    for col in ["Quantity", "NetPrice", "UnitPrice"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = load_data()

#clean dublicates

df = df.drop_duplicates()
# Basic cleaning & feature engineering

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# SalesRevenue = Quantity * UnitPrice
df["SalesRevenue"] = df["Quantity"] * df["UnitPrice"]

# Year, Month features
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)



with tab1:
    
        
    # Dashboard title
    st.title(" Transaction Data Dashboard ")
    st.header("General KPI Data")

    st.markdown("""
        <style>
        /* Sidebar container */
        [data-testid="stSidebar"] {
            background-color: #fff6f1; 
        }

        /* Sidebar yazıları */
        [data-testid="stSidebar"] * {
            color: black !important;   
        }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar filtreleri
    st.sidebar.header(" Filters")

    selected_category = st.sidebar.multiselect(
        "Select Category:", options=df["Category"].unique()
    )
    selected_city = st.sidebar.multiselect(
        "Select City:", options=df["City"].unique()
    )
    date_range = st.sidebar.date_input(
        "Data Range:",
        value=(df["InvoiceDate"].min(), df["InvoiceDate"].max()),
        min_value=df["InvoiceDate"].min(),
        max_value=df["InvoiceDate"].max(),
    )
    


    # Filtering
    df_filtered = df.copy()

    # Kategori filtresi
    if selected_category:  # liste boş değilse
        df_filtered = df_filtered[df_filtered["Category"].isin(selected_category)]

    # Şehir filtresi
    if selected_city:
        df_filtered = df_filtered[df_filtered["City"].isin(selected_city)]

    # Tarih filtresi (date_range her zaman dolu olacak)
    df_filtered = df_filtered[
        (df_filtered["InvoiceDate"] >= pd.to_datetime(date_range[0])) &
        (df_filtered["InvoiceDate"] <= pd.to_datetime(date_range[1]))
    ]
    # KPI list
    
    latest_year = df_filtered["Year"].max()
    prev_year = latest_year - 1

    current_year_data = df_filtered[df_filtered["Year"] == latest_year]
    prev_year_data = df_filtered[df_filtered["Year"] == prev_year]

    # KPI values
    sales_now = current_year_data["UnitPrice"].sum()
    sales_prev = prev_year_data["UnitPrice"].sum()
    netsales_now = current_year_data["NetPrice"].sum()
    netsales_prev = prev_year_data["NetPrice"].sum()
    orders_now = current_year_data["InvoiceNo"].nunique()
    orders_prev = prev_year_data["InvoiceNo"].nunique()
    cust_now = current_year_data["CustomerID"].nunique()
    cust_prev = prev_year_data["CustomerID"].nunique()

    # Change (%)
    sales_change = ((sales_now - sales_prev) / sales_prev * 100) if sales_prev > 0 else 0
    netsales_change = ((netsales_now - netsales_prev) / netsales_prev * 100) if netsales_prev > 0 else 0
    orders_change = ((orders_now - orders_prev) / orders_prev * 100) if orders_prev > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(" Total Sales", f"{sales_now:,.0f} TL", f"{sales_change:.1f}%")
    col2.metric("Total Net Sales", f"{netsales_now:,.0f} TL", f"{netsales_change:.1f}%")
    col3.metric(" Total Invoice", f"{orders_now:,}", f"{orders_change:.1f}%")
    col4.metric(" Total Customer Count", f"{cust_now:,}", f"{cust_prev:.1f}%")

    st.divider()

    # === Graphs ===

    col1, col2 = st.columns(2)
     # Monthly UnitPrice trends

    UnitPrice_over_time = df_filtered.groupby("Month")["UnitPrice"].sum().reset_index()
    fig_line = px.line(UnitPrice_over_time, x="Month", y="UnitPrice", markers=True, title="Monthly UnitPrice Trends", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_line.update_layout(
        paper_bgcolor=card_color,   
        plot_bgcolor=card_color,    
        font=dict(color="black"),   
        margin=dict(l=40, r=40, t=60, b=40)
    )
    col1.plotly_chart(fig_line, use_container_width=True)
    


    # Category based UnitPrice
   
    UnitPrice_by_category = df_filtered.groupby("Category")["UnitPrice"].sum().reset_index().sort_values("UnitPrice", ascending=False).head(10)
    fig_bar = px.bar(UnitPrice_by_category,
        x="Category", y="UnitPrice", color="Category",
        title="UnitPrice by Category", text_auto='.2s',color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_bar.update_xaxes(showticklabels=False)
    fig_bar.update_layout(bargap=0)
    fig_bar.update_layout(
        paper_bgcolor=card_color,   
        plot_bgcolor=card_color,    
        font=dict(color="black"),   
        margin=dict(l=40, r=40, t=60, b=40)
    )
    col2.plotly_chart(fig_bar, use_container_width=True)

   

with tab3:
    st.subheader("📦 ABC–XYZ Stock Classification")

    st.write(
        "ABC–XYZ analysis groups SKUs (StockCode) based on **revenue importance (ABC)** "
        "and **demand variability (XYZ)**. This helps inventory and supply chain decisions."
    )

    # --- Hocanın notebook mantığına paralel ---
    # 1) Month number from InvoiceDate
    df_abc = df_filtered.copy()
    df_abc["month_num"] = df_abc["InvoiceDate"].dt.month

    # 2) Monthly sales revenue per StockCode
    df_2 = (
        df_abc
        .groupby(["StockCode", "month_num"])["SalesRevenue"]
        .sum()
        .to_frame()
        .reset_index()
    )

    # 3) Pivot so that each month is a column
    df_3 = (
        df_2
        .pivot(index="StockCode", columns="month_num", values="SalesRevenue")
        .reset_index()
        .fillna(0)
    )

    # 4) Total sales, average monthly sales, standard deviation
    if df_3.shape[1] > 1:
        month_columns = df_3.columns[1:]  # all month columns
        df_3["total_sales"] = df_3[month_columns].sum(axis=1)
        df_3["average_sales"] = df_3["total_sales"] / len(month_columns)
        df_3["std_dev"] = df_3[month_columns].std(axis=1)

        # 5) Coefficient of variation (CV) = std / mean
        df_3["CV"] = np.where(
            df_3["average_sales"] > 0,
            df_3["std_dev"] / df_3["average_sales"],
            0.0
        )

        # 6) XYZ classification based on CV
        def xyz_analysis(x):
            if x <= 0.5:
                return "X"
            elif x > 0.5 and x <= 1:
                return "Y"
            else:
                return "Z"

        df_3["XYZ_Class"] = df_3["CV"].apply(xyz_analysis)
    # 7) ABC based on total revenue (sum of total_sales)
        df_4 = (
            df_3.groupby("StockCode")
            .agg(total_revenue=("total_sales", "sum"))
            .sort_values(by="total_revenue", ascending=False)
            .reset_index()
        )

        # Cumulative percentages
        df_4["cumulative"] = df_4["total_revenue"].cumsum()
        df_4["total_cumulative"] = df_4["total_revenue"].sum()
        df_4["sku_percent"] = df_4["cumulative"] / df_4["total_cumulative"]

        # ABC classification function
        def abc_classification(x):
            if x > 0 and x <= 0.80:
                return "A"
            elif x > 0.80 and x <= 0.95:
                return "B"
            else:
                return "C"

        df_4["ABC_Class"] = df_4["sku_percent"].apply(abc_classification)

        # 8) Merge ABC & XYZ info
        df_3_small = df_3[["StockCode", "total_sales", "average_sales", "std_dev", "CV", "XYZ_Class"]]
        df_4_small = df_4[["StockCode", "total_revenue", "sku_percent", "ABC_Class"]]

        df_final = df_4_small.merge(df_3_small, on="StockCode", how="left")

        # 9) Bring Description from original df
        df_desc = df_filtered[["StockCode", "Description"]].drop_duplicates()
        df_merge = df_final.merge(df_desc, on="StockCode", how="left")

        # 10) Remove duplicates and create final stock class
        df_result = df_merge.drop_duplicates().copy()
        df_result["stock_class"] = df_result["ABC_Class"].astype(str) + df_result["XYZ_Class"].astype(str)

        st.markdown("#### ABC–XYZ Summary Table")
        st.write(
            "Each row represents one **StockCode**, with its ABC and XYZ classes and key statistics."
        )
        st.dataframe(
            df_result[[
                "StockCode", "Description", "total_revenue",
                "ABC_Class", "XYZ_Class", "stock_class",
                "average_sales", "std_dev", "CV"
            ]].sort_values("total_revenue", ascending=False)
        )

        st.markdown("#### Stock Class Distribution (AX, BY, CZ, etc.)")
        class_counts = df_result["stock_class"].value_counts().reset_index()
        class_counts.columns = ["stock_class", "count"]

        fig_classes = px.bar(
            class_counts,
            x="stock_class",
            y="count",
            title="Number of SKUs in Each ABC–XYZ Class",
            text_auto=True
        )
        fig_line.update_layout(
            paper_bgcolor=card_color,   
            plot_bgcolor=card_color,    
            font=dict(color="black"),   
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_classes, use_container_width=True)

        st.markdown("#### Filter by Stock Class")
        selected_stock_class = st.selectbox(
                "Select a stock class (e.g. AX, BY, CZ)",
                options=sorted(df_result["stock_class"].unique())
            )
        filtered_df = df_result[df_result["stock_class"] == selected_stock_class]

        st.write(f"SKUs in class **{selected_stock_class}**:")
        st.dataframe(filtered_df[[
                "StockCode", "Description",
                "ABC_Class", "XYZ_Class",
                "total_revenue", "average_sales", "std_dev", "CV"
            ]])

    else:
        st.warning("Not enough data to perform ABC–XYZ analysis with current filters.")
    

with tab4:

    st.text("Please write the Description or the StockCode of the product you are looking for:")
    query = st.text_input("Product Description or StockCode")

    if query:
        # İçinde query geçen ürünleri bul (büyük/küçük harf duyarsız)
        mask = df["Description"].str.contains(query, case=False, na=False)
        results = df[mask]

        total_qty = results["Quantity"].sum()
        total_sales = results["NetPrice"].sum() 
        avg_price = results["UnitPrice"].mean()
        total_orders = results["InvoiceNo"].nunique()

        st.markdown("### 📦 Product Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Quantity", f"{total_qty:,}")
        c2.metric("Total Revenue (₺)", f"{total_sales:,.2f}")
        c3.metric("Average Unit Price (₺)", f"{avg_price:,.2f}")
        c4.metric("Number of Orders", f"{total_orders:,}")

        if results.empty:
            st.warning("No product found for this search.")
        else:
            st.success(f"{len(results)} product(s) found.")
            st.dataframe(results)


with tab2:
    
    st.subheader("📈 Descriptive Statistics")

    st.markdown("### Numeric Columns Summary")
    st.write(df_filtered[["Quantity", "NetPrice", "UnitPrice", "SalesRevenue"]].describe())


    st.subheader("Categorical Columns Summary")

    cat_cols = df.select_dtypes(include=["object"])

    summary = []

    for c in cat_cols.columns:
        summary.append({
            "Column": c,
            "Unique Count": df[c].nunique(),
            "Most Frequent": df[c].mode()[0],
            "Frequency": df[c].value_counts().iloc[0],
            "Percent (%)": round(df[c].value_counts(normalize=True).iloc[0] * 100, 2)
        })

    st.dataframe(pd.DataFrame(summary))
