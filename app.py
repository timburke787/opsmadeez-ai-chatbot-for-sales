import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import re

# Load environment variables
load_dotenv(dotenv_path=".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit config
st.set_page_config(page_title="CRM AI Assistant", layout="wide")
st.title("🤖 OpsMadeEZ | AI Buying Group Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ask a question input field
st.text_input(
    "Ask about a buying group (e.g., 'Who’s in the buying group for Acme Corp?')",
    key="user_question_input"
)

# Inject custom CSS for bubbles
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .chat-bubble-user, .chat-bubble-ai {
        max-width: 75%;
        padding: 0.75em 1em;
        border-radius: 15px;
        margin-bottom: 1em;
        line-height: 1.5;
    }
    .chat-bubble-user {
        align-self: flex-end;
        background-color: #5C33F6;
        color: white;
    }
    .chat-bubble-ai {
        align-self: flex-start;
        background-color: #E5E9F0;
        color: #3B3F5C;
    }
    .timestamp {
        font-size: 0.75em;
        margin-top: 4px;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# Chat history display
st.markdown("### Chat History")
for message in reversed(st.session_state.chat_history):
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

# Load CRM Data
@st.cache_data
def load_data():
    return {
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

data = load_data()

# Rename and prepare datasets
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

sales_activity_df = data["sales_activities"].rename(columns={
    "Contact ID": "contact_id",
    "Activity Type": "activity_type",
    "Date": "activity_date",
    "Summary": "summary"
})

roles_df = data["roles"].rename(columns={
    "Contact ID": "contact_id",
    "Opportunity ID": "opportunity_id",
    "Role": "role",
    "Is Primary": "is_primary"
})
roles_df["opportunity_id"] = roles_df["opportunity_id"].astype(str).str.strip()
deals_df["opportunity_id"] = deals_df["opportunity_id"].astype(str).str.strip()

# Merge to create buying group view
buying_group_df = roles_df.merge(contacts_df, on="contact_id", how="left")
buying_group_df = buying_group_df.merge(deals_df, on="opportunity_id", how="left")

# Normalize helper
def normalize(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

# Extract opportunity name
def extract_opportunity_name(question):
    norm_question = normalize(question)
    for _, row in accounts_df.iterrows():
        if normalize(row["account_name"]) in norm_question:
            acct_id = row["account_id"]
            matched = deals_df[deals_df["account_id"] == acct_id]
            if not matched.empty:
                return matched.iloc[0]["opportunity_name"]
    for opp in deals_df["opportunity_name"]:
        if normalize(opp) in norm_question:
            return opp
    return None

# ---------------------
# Run GPT if user input exists
# ---------------------
if st.session_state.get("user_question_input"):
    user_question = st.session_state["user_question_input"]
    opp_name = extract_opportunity_name(user_question or "")
    
    if opp_name:
        selected_group = buying_group_df[buying_group_df["opportunity_name"].str.lower() == opp_name.lower()]
        contact_ids = selected_group["contact_id"].unique()
        activity_subset = sales_activity_df[sales_activity_df["contact_id"].isin(contact_ids)]
        group_records = selected_group.to_dict(orient="records")
        activity_records = activity_subset.to_dict(orient="records")
    else:
        group_records = []
        activity_records = []

    prompt = f"""
You are an AI assistant helping a RevOps team analyze CRM data.

The user is asking a question about the buying group for an opportunity.

The buying group typically includes roles like Decision Maker, Champion, End User, Finance, and Procurement.

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

        response_text = response.choices[0].message.content
        timestamp = datetime.now().strftime("%b %d, %Y %I:%M %p")

        st.session_state.chat_history.append({
            "question": user_question,
            "answer": response_text,
            "timestamp": timestamp
        })

        st.session_state.user_question_input = ""  # Clear the input
        st.rerun()

    except Exception as e:
        st.error(f"Something went wrong: {e}")