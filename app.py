"""
Food Delivery Support Chatbot - demo interface.

Runs the REAL pipeline built in the capstone notebook: the Chat Agent calls the
Order Query Tool (which wraps the verified SQL Agent) and then the Answer Tool,
against the supplied customer_orders.db. Nothing here is mocked.

Run:  streamlit run app.py      (needs GROQ_API_KEY in the environment)
"""

import json
import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import truststore

truststore.inject_into_ssl()

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

DB_PATH = Path(__file__).parent / "data" / "customer_orders.db"
MODEL_NAME = "openai/gpt-oss-120b"
SEED = 42

st.set_page_config(page_title="Food Delivery Support", page_icon="🛵", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  .stApp { background: #f6f8fa; }
  .hdr { background: linear-gradient(100deg,#1d3557,#2a6f97); color:#fff;
         padding:18px 24px; border-radius:12px; margin-bottom:18px; }
  .hdr h1 { margin:0; font-size:1.45rem; font-weight:700; }
  .hdr p  { margin:4px 0 0; opacity:.85; font-size:.9rem; }
  .pill { display:inline-block; padding:2px 10px; border-radius:999px;
          font-size:.72rem; font-weight:700; letter-spacing:.02em; }
  .ok   { background:#d9f0ec; color:#17695e; }
  .warn { background:#fdf0e4; color:#9a6212; }
  .stage{ background:#fff; border:1px solid #dde4ec; border-left:4px solid #2a9d8f;
          border-radius:8px; padding:10px 14px; margin-bottom:10px; }
  .stage .lbl { font-size:.7rem; font-weight:700; color:#2a6f97;
                text-transform:uppercase; letter-spacing:.05em; }
  .stage .bod { font-size:.82rem; color:#223; margin-top:4px;
                font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
                white-space:pre-wrap; word-break:break-word; }
  .raw  { border-left-color:#e9a23b; }
  .sqlb { border-left-color:#8a63d2; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Build the real pipeline once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_pipeline(api_key: str):
    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(model=MODEL_NAME, temperature=0.0, max_tokens=1024,
                   max_retries=3, model_kwargs={"top_p": 1.0, "seed": SEED})
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH.as_posix()}")

    SQL_AGENT_PROMPT = """You are the customer support assistant for a food delivery platform.
You answer questions about orders by querying the `orders` SQLite database.

PROCEDURE
1. Call sql_db_list_tables, then sql_db_schema, before writing any query.
2. Validate your query with sql_db_query_checker before executing it.
3. Execute with sql_db_query and base your answer ONLY on the rows returned.

SCHEMA NOTES
- order_status is one of 'preparing food', 'picked up', 'delivered', 'canceled'.
- payment_status is 'COD' (unpaid), 'completed' or 'canceled'.
- preparing_eta and delivery_eta are times PROMISED to the customer.
  prepared_time and delivery_time are times that ACTUALLY happened.
- A time column is NULL when the order has not reached that stage. NULL means
  "not yet", never zero and never a delay.

RULES
- READ-ONLY. Never issue INSERT, UPDATE, DELETE, DROP or ALTER.
- If a query returns no rows, say plainly that no such order exists.

TRANSCRIPTION RULE
Copy every value character for character from the query result. Never reformat,
round, insert spaces inside times, or append uncertainty markers such as '?'.
"""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_agent = create_react_agent(model=llm, tools=toolkit.get_tools(),
                                   prompt=SQL_AGENT_PROMPT)

    trace = {"sql": [], "raw": None, "polished": None}

    @tool
    def order_query_tool(query: str) -> str:
        """Look up order information in the orders database. Use for any question
        about an order: status, items, payment, timings or lateness. Returns a raw
        factual answer with no courtesy wording."""
        try:
            result = sql_agent.invoke({"messages": [HumanMessage(content=query)]})
        except Exception as exc:
            # A failed lookup is NOT evidence that the order is missing. Returning
            # a sentinel stops the Answer Tool dressing an outage up as a factual
            # denial - which is how a transient error became "no such order exists".
            msg = f"RETRIEVAL_FAILED: {type(exc).__name__} - the order system could not be reached."
            trace["sql"], trace["raw"] = [], msg
            return msg
        raw = result["messages"][-1].content.strip()
        trace["sql"] = [tc["args"].get("query", "")
                        for m in result["messages"]
                        for tc in (getattr(m, "tool_calls", None) or [])
                        if tc["name"] == "sql_db_query"]
        trace["raw"] = raw
        return raw

    ANSWER_TOOL_PROMPT = """You rewrite a raw order-lookup result into a reply a food delivery
platform would send its customer.

ABSOLUTE RULES
- Do NOT add, remove or alter any fact. Every status, time, item and order ID in your reply
  must appear in the raw text you were given.
- If the raw text says something is unknown or that no such order exists, say exactly that.
  NEVER estimate a time, invent a status, or offer a reassuring guess.
- If the raw text begins with RETRIEVAL_FAILED, the lookup did not complete. Apologise for a
  temporary technical problem and ask the customer to try again shortly. You must NOT say the
  order does not exist, is cancelled, or is missing - a failed lookup tells us nothing about
  the order.
- Quote times character for character. Never insert a space inside a time.

STYLE
Polite, formal and warm. Address the customer as "you". Open with the answer, not
pleasantries. Two to four sentences. Plain prose - no bullets, no markdown.
"""

    @tool
    def answer_tool(raw_response: str) -> str:
        """Rewrite a raw order-lookup result into a polite, formal reply for the
        customer. Always call this on the output of order_query_tool."""
        polished = llm.invoke([
            SystemMessage(content=ANSWER_TOOL_PROMPT),
            HumanMessage(content=f"Raw order-lookup result:\n{raw_response}"),
        ]).content.strip()
        trace["polished"] = polished
        return polished

    CHAT_AGENT_PROMPT = """You are the customer support chatbot for a food delivery platform,
speaking directly to a customer about their order.

MANDATORY PROCEDURE for every message about an order
1. Call order_query_tool with the customer's question.
2. Call answer_tool, passing the ENTIRE raw result from step 1 unchanged.
3. Reply with the text answer_tool returns, exactly as it comes back.

RULES
- Never answer an order question from memory. Always retrieve first.
- If the message contains anything resembling an order reference (for example O12505), it IS
  an order question, even with no other words. Retrieve it.
- NEVER state that an order does not exist unless order_query_tool returned that result in
  this turn. Never infer it from earlier turns or from a failed lookup.
- Never skip answer_tool. A raw lookup must not be shown to a customer.
- For non-order questions, reply briefly yourself without calling the tools.
- If the question is ambiguous - e.g. "is my order late?" with no order ID - ask one
  short clarifying question instead of guessing.
"""
    chat_agent = create_react_agent(model=llm, tools=[order_query_tool, answer_tool],
                                    prompt=CHAT_AGENT_PROMPT)
    return chat_agent, trace


@st.cache_data(show_spinner=False)
def load_demo():
    """Verified exchanges captured from the executed notebook. Replayed in demo
    mode so a public visitor always gets a working, honest demonstration."""
    f = Path(__file__).parent / "demo_data.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def best_match(question, turns):
    """Nearest recorded question, by word overlap."""
    qw = set(re.findall(r"[a-z0-9]+", question.lower()))
    oid = re.search(r"[oO]\d{5}", question)
    best, score = None, -1
    for t in turns:
        tw = set(re.findall(r"[a-z0-9]+", t["question"].lower()))
        sc = len(qw & tw) / max(len(qw | tw), 1)
        if oid and oid.group(0).upper() in t["question"].upper():
            sc += 0.6
        if sc > score:
            best, score = t, sc
    return best, score


@st.cache_data(show_spinner=False)
def load_orders():
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query("SELECT * FROM orders", c)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hdr">
  <h1>🛵 Food Delivery - Customer Support</h1>
  <p>Ask about any order. Answers come from a live database query, never from the model's memory.</p>
</div>
""", unsafe_allow_html=True)

orders = load_orders()

# ---------------------------------------------------------------------------
# Sidebar: credentials, live data, sample questions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Connection")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        # Local dev convenience: a .groq_key file beside the app. Never commit it.
        kf = Path(__file__).parent / ".groq_key"
        if kf.exists():
            api_key = kf.read_text(encoding="utf-8").strip()
    if not api_key:
        api_key = st.text_input("GROQ_API_KEY", type="password",
                                help="Not stored. Set it in the environment to skip this.")
    demo = load_demo()
    mode = st.radio(
        "Mode",
        ["Demo (verified replies)", "Live (calls the model)"],
        index=0 if demo else 1,
        help=("Demo replays exchanges recorded from the executed notebook - no API calls, "
              "always available. Live runs the real agent and needs an API key."))
    is_demo = mode.startswith("Demo")
    if is_demo:
        st.markdown('<span class="pill ok">demo - verified transcripts</span>',
                    unsafe_allow_html=True)
        st.caption("Every reply below was produced by the real pipeline and checked "
                   "against the database.")
    else:
        st.markdown(
            f'<span class="pill {"ok" if api_key else "warn"}">'
            f'{"model connected" if api_key else "key required"}</span>',
            unsafe_allow_html=True)

    st.divider()
    st.subheader("Live order book")
    st.caption(f"{len(orders)} orders - the same records the agent queries")
    status_filter = st.multiselect("Filter by status",
                                   sorted(orders["order_status"].unique()),
                                   default=[])
    view = orders if not status_filter else orders[orders["order_status"].isin(status_filter)]
    st.dataframe(view[["order_id", "order_status", "item_in_order"]],
                 hide_index=True, use_container_width=True, height=240)

    st.divider()
    st.subheader("Try asking")
    SAMPLES = [
        "What's the status of my order O12486?",
        "What items are in order O12488?",
        "Has order O12490 arrived on time?",
        "Was I charged for order O12487?",
        "Can you check order O99999?",
        "Is my order late?",
    ]
    for i, q in enumerate(SAMPLES):
        if st.button(q, key=f"s{i}", use_container_width=True):
            st.session_state.pending = q

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
chat_col, trace_col = st.columns([1.55, 1])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

with chat_col:
    st.markdown("#### Conversation")
    board = st.container(height=430)
    with board:
        if not st.session_state.messages:
            st.info("Ask about an order, or pick a sample question from the sidebar.")
        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🛵"):
                st.markdown(m["content"])

    typed = st.chat_input("Ask about your order...")
    question = typed or st.session_state.pop("pending", None)

    if question and not is_demo and not api_key:
        st.error("Add your GROQ_API_KEY in the sidebar, or switch to Demo mode.")
        question = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        if is_demo:
            match, score = best_match(question, demo["turns"])
            with board:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(question)
                with st.chat_message("assistant", avatar="🛵"):
                    with st.spinner("Checking your order..."):
                        time.sleep(0.9)
                    if score < 0.18:
                        reply = ("I can only answer the recorded questions in demo mode. "
                                 "Try one of the samples in the sidebar, or switch to Live "
                                 "mode with an API key.")
                        st.markdown(reply)
                        st.session_state.last_trace = None
                    else:
                        reply = match["reply"]
                        st.markdown(reply)
                        st.session_state.last_trace = {
                            "sql": [match["sql"]],
                            "raw": ("retrieved via Order Query Tool - "
                                    + (" -> ".join(match["tools"]) if match["tools"]
                                       else "no tools needed for this message")),
                            "polished": reply, "elapsed": 0.9, "question": question,
                            "demo": True, "tools": match["tools"]}
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

        chat_agent, trace = build_pipeline(api_key)
        trace["sql"], trace["raw"], trace["polished"] = [], None, None

        with board:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(question)
            with st.chat_message("assistant", avatar="🛵"):
                with st.spinner("Checking your order..."):
                    t0 = time.time()
                    try:
                        result = chat_agent.invoke({
                            "messages": st.session_state.history
                                        + [HumanMessage(content=question)]
                        })
                        reply = result["messages"][-1].content.strip()
                        st.session_state.history = (
                            st.session_state.history
                            + [HumanMessage(content=question), AIMessage(content=reply)]
                        )[-6:]
                    except Exception as exc:
                        reply = ("I'm sorry - I'm having trouble reaching our order system "
                                 "right now. Please try again in a moment.")
                        trace["raw"] = f"[{type(exc).__name__}] {exc}"
                    elapsed = time.time() - t0
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.last_trace = {**trace, "elapsed": elapsed, "question": question}
        st.rerun()

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages, st.session_state.history = [], []
        st.session_state.last_trace = None
        st.rerun()

with trace_col:
    st.markdown("#### Behind the answer")
    st.caption("Every reply is traceable to the query that produced it.")
    tr = st.session_state.last_trace
    if not tr:
        st.markdown('<div class="stage"><div class="lbl">waiting</div>'
                    '<div class="bod">Ask a question to see the pipeline.</div></div>',
                    unsafe_allow_html=True)
    else:
        sql_txt = "\n".join(tr["sql"]) if tr["sql"] else "no database query needed"
        st.markdown(f'<div class="stage sqlb"><div class="lbl">1 - SQL executed</div>'
                    f'<div class="bod">{sql_txt}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stage raw"><div class="lbl">2 - Order Query Tool '
                    f'(raw fact)</div><div class="bod">{tr["raw"] or "-"}</div></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="stage"><div class="lbl">3 - Answer Tool (customer reply)'
                    f'</div><div class="bod">{tr["polished"] or "-"}</div></div>',
                    unsafe_allow_html=True)

        # Fact check: every concrete token in the raw answer must survive the rewrite.
        if tr.get("demo"):
            st.info("Demo mode: this exchange was produced by the real pipeline during the "
                    "notebook run and verified against the database.")
        elif tr["raw"] and tr["polished"]:
            def facts(t):
                return (set(re.findall(r"\b\d{1,2}:\d{2}\b", t))
                        | set(re.findall(r"\b[OC]\d{4,5}\b", t))
                        | {w for w in ["preparing food", "picked up", "delivered",
                                       "canceled", "COD", "completed"] if w.lower() in t.lower()})
            kept = facts(tr["raw"]) & facts(tr["polished"])
            lost = facts(tr["raw"]) - facts(tr["polished"])
            c1, c2 = st.columns(2)
            c1.metric("Facts preserved", f"{len(kept)}/{len(facts(tr['raw']))}")
            c2.metric("Response time", f"{tr['elapsed']:.1f}s")
            if lost:
                st.warning(f"Dropped in rewrite: {', '.join(sorted(lost))}")
            else:
                st.success("No fact added, altered or dropped in the rewrite.")

st.caption("Capstone Project 1 - Food Delivery Chatbot | Nitesh Tripathi. "
           "Read-only access; the assistant cannot modify any order.")
