import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up the page
st.set_page_config(page_title="CRM AI Assistant", layout="wide")
st.title("🤖 CRM Buying Group Assistant")

# Load CRM data from CSV files
@st.cache_data
def load_data():
    data = {
        "contacts": pd.read_csv("data/contacts.csv"),
        "accounts": pd.read_csv("data/accounts.csv"),
        "deals": pd.read_csv("data/deals.csv"),
        "sales_activities": pd.read_csv("data/sales_activities.csv"),
        "marketing": pd.read_csv("data/marketing_touchpoints.csv"),
        "contact_funnel": pd.read_csv("data/contact_funnel_history.csv"),
        "deal_funnel": pd.read_csv("data/deal_funnel_history.csv"),
        "roles": pd.read_csv("data/contact_deal_roles.csv"),
        "definitions": pd.read_csv("data/buying_group_definitions.csv")
    }
    return data

data = load_data()

# Ask the assistant a question
user_question = st.text_input("Ask a question about your CRM data:")

# ---------------------
# Rename contact fields
contacts_df = data["contacts"].rename(columns={
    "Contact ID": "contact_id",
    "Full Name": "full_name",
    "Email": "email",
    "Title": "title",
    "Phone": "phone",
    "Location": "location",
    "Last Engagement Date": "last_engagement_date",
    "Engagement Score": "engagement_score",
    "Account ID": "account_id"
})

accounts_df = data["accounts"].rename(columns={
    "Account ID": "account_id",
    "Company Name": "account_name",  # ✅ Corrected
    "Industry": "industry",
    "NAICS Code": "sic_naics",
    "Region": "region",
    "Domain": "domain",
    "Employee Count": "employee_count",
    "Annual Revenue": "annual_revenue",
    "Industry Name": "industry_name"
})

# Rename deal fields
deals_df = data["deals"].rename(columns={
    "Opportunity ID": "opportunity_id",
    "Opportunity Name": "opportunity_name",
    "Stage": "stage",
    "Type": "type",
    "Amount": "amount",
    "Created Date": "created_date",
    "Expected Close Date": "expected_close_date",
    "Account ID": "account_id",
    "Primary Contact ID": "primary_contact_id",
    "Primary Contact Name": "primary_contact_name",
    "Primary Contact Title": "primary_contact_title"
})

# Rename sales activity fields
sales_activity_df = data["sales_activities"].rename(columns={
    "Contact ID": "contact_id",
    "Activity Type": "activity_type",
    "Date": "activity_date",
    "Summary": "summary"
})

# Rename contact-opportunity role fields
buying_group_roles = data["roles"].rename(columns={
    "Contact ID": "contact_id",
    "Opportunity ID": "opportunity_id",
    "Role": "role",
    "Is Primary": "is_primary"
})

# Merge to create full buying group view
buying_group_df = buying_group_roles.merge(contacts_df, on="contact_id").merge(deals_df, on="opportunity_id")

# ---------------------
# Match opportunity based on account name
# ---------------------
def normalize(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

def extract_opportunity_name(question):
    norm_question = normalize(question)

    # Try to match based on account name
    for i, row in accounts_df.iterrows():
        if normalize(row["account_name"]) in norm_question:
            acct_id = row["account_id"]
            matched_opps = deals_df[deals_df["account_id"] == acct_id]
            if not matched_opps.empty:
                return matched_opps.iloc[0]["opportunity_name"]

    # Fallback: match based on full opportunity name
    for opp in deals_df["opportunity_name"]:
        if normalize(opp) in norm_question:
            return opp

    return None

opp_name = extract_opportunity_name(user_question or "")

# ---------------------
# Filter records for selected opportunity
# ---------------------
if opp_name:
    selected_group = buying_group_df[buying_group_df["opportunity_name"].str.lower() == opp_name.lower()]
    contact_ids = selected_group["contact_id"].unique()
    activity_subset = sales_activity_df[sales_activity_df["contact_id"].isin(contact_ids)]

    group_records = selected_group.to_dict(orient="records")
    activity_records = activity_subset.to_dict(orient="records")
else:
    group_records = []
    activity_records = []

# Optional: Debug helpers
# st.write("📊 sales_activity_df columns:", sales_activity_df.columns.tolist())
# st.write("🧾 Available opportunity names:", deals_df["opportunity_name"].tolist())
# st.write("🔍 Matched opportunity:", opp_name)

# ---------------------
# Prompt GPT
# ---------------------
if user_question:
    with st.spinner("Thinking..."):
        prompt = f"""
You are an AI assistant helping a RevOps team analyze CRM data.

The user is asking a question about the buying group for an opportunity.

The buying group typically includes the following roles:
- Decision Maker (e.g. CMO, VP of Marketing)
- Champion (someone who drives adoption internally)
- End User (daily users of the product)
- Finance (budget holder)
- Procurement (contract gatekeeper)

Your goals:
1. Identify which of those roles are represented in the buying group and which are missing.
2. Review the sales activity history to identify:
   - The most engaged contact (based on activity frequency and recency)
   - The least engaged contact
   - Any contacts who haven't been touched recently
   - Summaries of the last few activities if available

Here is the buying group for the opportunity '{opp_name}' (if found):
{group_records}

Here are the sales activities involving those contacts:
{activity_records}

Now, based on the question below and the data above, provide an analysis or answer:

{user_question}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful CRM and RevOps assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            st.markdown("### 🧠 AI Response")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

