import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv(dotenv_path=".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up the page

st.set_page_config(page_title="CRM AI Assistant", layout="wide")

st.title("🤖 OpsMadeEZ | AI Buying Group Assistant")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# ✅ Inject custom styles for chat bubbles and layout
st.markdown("""
    <style>
    .chat-bubble-user {
        background-color: #5C33F6;
        color: white;
        padding: 0.75em 1em;
        border-radius: 15px;
        margin-bottom: 0.5em;
        max-width: 80%;
        align-self: flex-end;
    }
    .chat-bubble-ai {
        background-color: #E5E9F0;
        color: #3B3F5C;
        padding: 0.75em 1em;
        border-radius: 15px;
        margin-bottom: 1em;
        max-width: 80%;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .timestamp {
        font-size: 0.75em;
        margin-top: 4px;
        display: block;
        color: #9CA3AF;
    }
    </style>
""", unsafe_allow_html=True)
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
# Ask the assistant a question inside a form
with st.form("user_input_form"):
    user_question = st.text_input("Ask about a buying group (use the deal names mentioned above):", key="user_question_input")
    submitted = st.form_submit_button("Submit")

# Only run the assistant if the form was submitted and question exists
if submitted and user_question:
    # Extract opportunity name
    opp_name = extract_opportunity_name(user_question or "")

    # Filter records
    if opp_name:
        selected_group = buying_group_df[buying_group_df["opportunity_name"].str.lower() == opp_name.lower()]
        contact_ids = selected_group["contact_id"].unique()
        activity_subset = sales_activity_df[sales_activity_df["contact_id"].isin(contact_ids)]
        group_records = selected_group.to_dict(orient="records")
        activity_records = activity_subset.to_dict(orient="records")
    else:
        group_records = []
        activity_records = []
# Prompt GPT
# ---------------------
if st.session_state.get("user_question_input"):
    user_question = st.session_state["user_question_input"]
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

# Render chat history
st.markdown("### Chat History")
chat_history_reversed = list(reversed(st.session_state.chat_history))

for message in chat_history_reversed:
    st.markdown(f""" 
    <div class="chat-container">
        <div class="chat-bubble-user">
            <strong>You:</strong> {message['question']}
            <span class="timestamp">{message['timestamp']}</span>
        </div>
        <div class="chat-bubble-ai">
            <strong>AI:</strong> {message['answer']}
            <span class="timestamp">{message['timestamp']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
    "Company Name": "account_name",
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

# Normalize IDs before join
roles_df = data["roles"].rename(columns={
    "Contact ID": "contact_id",
    "Opportunity ID": "opportunity_id",
    "Role": "role",
    "Is Primary": "is_primary"
})
roles_df["opportunity_id"] = roles_df["opportunity_id"].astype(str).str.strip()
deals_df["opportunity_id"] = deals_df["opportunity_id"].astype(str).str.strip()

# Merge to create full buying group view
buying_group_df = roles_df.merge(contacts_df, on="contact_id", how="left")
buying_group_df = buying_group_df.merge(deals_df, on="opportunity_id", how="left")
valid_opps = buying_group_df["opportunity_name"].dropna().unique().tolist()
# Show intro + available deals only after data is loaded
st.markdown(f"""
Welcome to the **OpsMadeEZ CRM Buying Group Assistant**, built by Tim Burke.

This AI-powered chatbot helps sellers, marketers, and RevOps teams explore CRM data and make better decisions about active opportunities and their buying groups.

Try asking high-value questions like:
- “Who is in the buying group for Rogers-Wilson?”
- “What roles are missing from the buying group for the Rivera-Ho deal?”
- “Which contact is the most engaged on the Dickerson-Medina deal?”

---

### 📋 Opportunities with Buying Group Members

Use these opportunity names in your questions:
{', '.join(sorted(valid_opps))}
""")
# --------------------
# Rename sales activity fields
sales_activity_df = data["sales_activities"].rename(columns={
    "Contact ID": "contact_id",
    "Activity Type": "activity_type",
    "Date": "activity_date",
    "Summary": "summary"
})

# ---------------------
# Match opportunity based on account name
# ---------------------
def normalize(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

def extract_opportunity_name(question):
    norm_question = normalize(question)

    for _, row in accounts_df.iterrows():
        if normalize(row["account_name"]) in norm_question:
            acct_id = row["account_id"]
            matched_opps = deals_df[deals_df["account_id"] == acct_id]
            if not matched_opps.empty:
                return matched_opps.iloc[0]["opportunity_name"]

    for opp in deals_df["opportunity_name"]:
        if normalize(opp) in norm_question:
            return opp

    return None

# Extract opportunity name
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

# ---------------------
try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful CRM and RevOps assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = response.choices[0].message.content

            from datetime import datetime
            timestamp = datetime.now().strftime("%b %d, %Y %I:%M %p")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            st.session_state.chat_history.append({
                "question": user_question,
                "answer": response_text,
                "timestamp": timestamp
            })
            st.session_state.submitted = True
            st.rerun()

except Exception as e:
            st.error(f"Something went wrong: {e}")      