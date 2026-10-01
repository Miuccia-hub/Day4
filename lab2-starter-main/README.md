# LAWS90286 - LAB 2

## Completed tool

This starter has been implemented as **Canvassian DealScope**, a local,
evidence-grounded M&A due-diligence application. It ingests every supplied text
document, creates overlapping chunks with metadata, stores deterministic local
embeddings in ChromaDB, reranks search results for exact legal language, reviews
change-in-control terms for the six major clients, and generates a cited Board
report. No confidential document content is sent to an external model.

### Run locally

Python 3.11 or 3.12 is recommended for the widest ChromaDB compatibility.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The first run builds `.chroma/`; subsequent runs reuse it unless the source
files change. Useful command-line equivalents are:

```bash
python cli.py index
python cli.py search "PayWise financial difficulty unpaid invoices"
python cli.py report
```

The report is written to `outputs/canvassian_due_diligence_report.md`. If
ChromaDB is temporarily unavailable, the same workflow uses an in-memory local
search backend, so `python cli.py ...` remains usable (the Streamlit UI still
requires Streamlit).

### Verification

```bash
python -m unittest discover -s tests
```

The supplied archive contains 750 documents (50 board papers, 100 contracts and
600 emails), rather than the 1,000 documents described below. The application
surfaces this completeness limitation in the UI and generated report.

## Instructions

#### Objective

To be able to rapidly search a large volume of evidence to look for critical insights and to generate a report. We will assume that the evidence takes the form of documents and the context is a mergers and acquisitions (M&A) transaction.

We act for the purchaser. Our client is looking to acquire a software company (Canvassian Pty Ltd). The company sells cybersecurity software.

#### Risks

As the purchaser, out client is concerned about any liabilities or risks that they haven't accounted for properly. Ultimately, these are things that could mean that the company is worth less than the price Canvassian is considering paying for it.

We have been instructed that the following considerations are critical:

- the founder of Canvassian (Jane Wu) has inspirationally led the company's success, and her continued involvement and motivation to lead the company is regarded as vital to the deal being successful

- there are rumours that Canvassian's largest client (PayWise Pty Ltd) is having financial difficulties. The PayWise account comprises 20% of Canvassian's revenue, so trying to verify these rumours and their potential impact is material to the proposed price.

- in addition to PayWise, there are another 5 clients who together account for 40% of Canvassian's revenue (the remaining 40% is spread of hundreds of smaller clients). For PayWise and these other 5 clients (Alphabear, Bravocat, Charlemont, Deltaforce and Echona), their contractual arrangements need to be reviewed to confirm that there are no "change in control" terms that might be adverse to Canvassian's interests in purchasing Canvassian.

In addition to these critical risks, as an expert deal-maker you should make sure to look for any other risks that you think might be relevant.

#### Timeframe

The timeframe for the deal is urgent - the Board needs your opinion this afternoon. You don't have the time to do a manual due diligence, looking at every document. What you need to do is to quickly extract meaningful insights from the documents - reporting on each of the critical risks as well as any other significant risks that you come across.

#### Output

The client needs a report that they can present to their Board for sign-off on the deal.

## Technical Details

You have 1000 documents, which you know includes:

- emails
- contracts
- board papers

These are provided to you in a zip file. They are all in plaintext format - your IT team has preprocessed them for you.

## System Design

1. Create a ChromaDB vector database

2. Add all documents to the vector database
    - read the documents
    - extract any necessary metadata from the documents
    - chunk the documents

3. Create a retrieval pipeline to find the most relevant chunks corresponding to different queries (consider using an AI agent that has the ability to come up with vectordb queries)

4. Prescribe how to handle the returned chunks to create a meaningful response

5. Generate a written report for your client, based on the responses

## ChromaDB & SQLite3

There is an issue with ChromaDB, where to run it locally requires a newer version of sqlite3 (a ddatabase software) than comes out-of-the-box with both Codespaces and Streamlit Community Cloud. Fortunately, there are fixes:

To use ChromaDB in Codespaces:

> conda create -n . python=3.12

> conda init

> conda activate .

> conda install -c conda-forge sqlite

> pip3 install -r requirements.txt

To use ChromaDB in Streamlit Community Cloud:

1. update requirements.txt

```
protobuf==3.20.1
pysqlite3-binary
```

2. point the application to the updated version of sqlite3, in Home.py

```
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
```

## Notes

#### Axios Error?

If you are getting an axios error on document upload, do the following:

1. Create a new directory: ".streamlit"

2. Create a new file under ./streamlit: "config.toml"

3. Edit config.toml as:

```
[server]
enableXsrfProtection = false
enableCORS = false
```

#### Llama-Index References

- https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/

- https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/
