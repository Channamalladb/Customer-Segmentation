import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.markdown("<style>...</style>", unsafe_allow_html=True)


# Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    df.rename(columns={
        "Annual Income (k$)": "Income",
        "Spending Score (1-100)": "Score"
    }, inplace=True)
    return df

df = load_data()

# Sidebar controls
st.sidebar.title("Segmentation Controls")
k = st.sidebar.slider("Number of Segments (k)", 2, 7, 5)

# Title
st.title("🧠 Interactive Customer Segmentation")
st.markdown("""
This dashboard analyzes mall customer data to identify distinct segments and inform marketing strategy.
""")

# Overview metrics
st.header("Customer Base Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", df.shape[0])
col2.metric("Avg. Age", f"{df['Age'].mean():.1f}")
col3.metric("Avg. Income (k$)", f"{df['Income'].mean():.1f}")
col4.metric("Avg. Score", f"{df['Score'].mean():.1f}")

# Gender Distribution
st.subheader("Gender Distribution")
gender_counts = df['Gender'].value_counts()
fig, ax = plt.subplots()
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# Age Distribution
st.subheader("Age Distribution")
fig, ax = plt.subplots()
sns.histplot(df['Age'], bins=10, kde=True, ax=ax)
ax.set_xlabel("Age")
ax.set_ylabel("Count")
st.pyplot(fig)

# K-Means Clustering
st.header("Segmentation Analysis Lab")
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Income', 'Score']])

# Scatter Plot by Income vs Score
st.subheader("Customer Segments")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="Income", y="Score", hue="Cluster", palette="tab10", ax=ax)
ax.set_title("K-Means Clustering")
st.pyplot(fig)

# Cluster Profiles
st.subheader("Cluster Profiles")
cluster_summary = df.groupby('Cluster').agg({
    'Income': 'mean',
    'Score': 'mean',
    'Age': 'mean'
}).round(1).reset_index()
st.dataframe(cluster_summary)

# Strategic Insights
st.header("Strategic Insights")
selected_cluster = st.selectbox("Select Segment for Strategy", cluster_summary['Cluster'])
profile = cluster_summary[cluster_summary['Cluster'] == selected_cluster].iloc[0]

st.markdown(f"**Persona Name**: Segment {selected_cluster}")
st.markdown(f"""
- **Average Age**: {profile['Age']} years  
- **Average Income**: ${profile['Income']}k  
- **Average Spending Score**: {profile['Score']}
""")

# Predefined strategy suggestions
strategy_map = {
    0: "🎯 High-income, high-spending: Promote exclusive memberships, VIP events, and premium bundles.",
    1: "🛍️ Low-income, high-spending: Emphasize value-for-money, limited-time offers, and flash sales.",
    2: "💡 High-income, low-spending: Educate on brand value and introduce trial incentives.",
    3: "🎁 Young age group with medium income: Highlight lifestyle branding and seasonal fashion.",
    4: "🧧 Budget-conscious, low-spending: Leverage referral programs, and budget bundle packs.",
    5: "📣 Digital-savvy with variable spend: Focus on influencer marketing and app-exclusive deals.",
    6: "📌 Niche or mixed group: Use exploratory campaigns and A/B tested promotions."
}

strategy = strategy_map.get(selected_cluster, "📌 Consider deeper profiling for this segment.")
st.info(f"🧠 Suggested Strategy: {strategy}")

# Data Explorer
st.header("Data Explorer")
with st.expander("View Full Dataset"):
    st.dataframe(df)

if st.button("Generate Custom Data Summary"):
    avg_age = df['Age'].mean()
    avg_income = df['Income'].mean()
    avg_score = df['Score'].mean()
    gender_ratio = df['Gender'].value_counts(normalize=True) * 100

    st.markdown("### 📊 Summary Insights")
    st.markdown(f"""
    - The dataset includes {df.shape[0]} customers.
    - Average age is approximately **{avg_age:.1f}** years.
    - Average annual income is **${avg_income:.1f}k**.
    - Average spending score is **{avg_score:.1f}**.
    - Gender distribution: **{gender_ratio.get('Female', 0):.1f}% Female**, **{gender_ratio.get('Male', 0):.1f}% Male**.
    """)
# Footer
st.markdown("---")      
st.markdown("© 2023 Customer Segmentation Dashboard. All rights reserved.")
# Add custom CSS for styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .stDataFrame th {
        background-color: #f2f2f2;
        color: #333;
    }
    .stDataFrame td {
        color: #555;
    }
    .stMarkdown {
        font-size: 16px;
        line-height: 1.6;
    }
    .stHeader {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .stSubheader {
        font-size: 20px;
        color: #555;
    }
    .stMetric {
        font-size: 18px;
        color: #333;
    }
    .stInfo {
        background-color: #e7f3fe;
        color: #31708f;
        border-left: 4px solid #31708f;
        padding: 10px;
        border-radius: 5px;
    }
    .stAlert {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 4px solid #f5c6cb;
        padding: 10px;
        border-radius: 5px;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-left: 4px solid #c3e6cb;
        padding: 10px;
        border-radius: 5px;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border-left: 4px solid #ffeeba;
        padding: 10px;
        border-radius: 5px;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 4px solid #f5c6cb;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)
