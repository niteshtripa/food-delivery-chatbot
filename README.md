# Food Delivery Support Chatbot

A customer-support assistant that answers order questions **from a live database query**,
never from the language model's memory.

Built as Capstone Project 1 for the PG Programme in AI & Machine Learning.

## What it does

Ask about an order — status, items, payment, whether it arrived on time — and the
assistant answers from the order database. The panel on the right shows the SQL that
produced the answer, so every reply is traceable.

## How it works

The architecture separates **getting the fact right** from **saying it well**, because
those two goals pull against each other:

| Stage | Responsibility |
|---|---|
| SQL Agent | Discovers the schema, validates and runs a read-only query |
| Order Query Tool | Wraps the SQL Agent, returns the raw fact with no courtesy wording |
| Answer Tool | Rewrites that fact into a polite reply — with no database access, so it *cannot* embellish |
| Chat Agent | Orchestrates the two, in that order, every time |

Removing the Answer Tool's database access is the key decision: it is a stronger
guarantee than instructing a capable component to behave.

## Verified, not asserted

Every result was checked against independently computed SQL ground truth:

- **10/10** fields correct when retrieving a complete order record
- **6/6** agent tests across lookup, grouping, missing-value semantics and ranking
- **8/8** chatbot replies, including a non-existent order (refused, not fabricated)
  and an ambiguous question (clarified, not guessed)

## Running it

    pip install -r requirements.txt
    streamlit run app.py

**Demo mode** (default) replays verified exchanges recorded from the executed
notebook — no API calls, always available.

**Live mode** runs the real agent and needs a `GROQ_API_KEY`, set as an environment
variable or in Streamlit secrets. Never commit the key.

## Data

`customer_orders.db` — 20 synthetic order records supplied with the coursework. No real
customer data is used anywhere in this project.
