from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

client = create_client(url, key)

companies = [
    ("coforge.com", "dataset002"),
    ("bpc.fund", "dataset002"),
    ("pathlabs.com", "dataset002"),
    ("zadv.com", "dataset002"),
    ("dagger.agency", "dataset002"),
    ("narrativedc.com", "dataset002"),
    ("bounteous.com", "dataset002"),
    ("ramapo.edu", "dataset002"),
    ("keypointintelligence.com", "dataset002"),
    ("jubilantweb.com", "dataset002"),
    ("stridehealth.com", "dataset002"),
    ("princetoninternetmarketing.com", "dataset002"),
    ("colonyspark.com", "dataset002"),
    ("toolboxstudios.com", "dataset002"),
    ("heraldpr.com", "dataset002"),
    ("millionlabs.co.uk", "dataset002"),
    ("publicis.com", "dataset002"),
    ("hss.edu", "dataset002"),
]

for domain, dataset_id in companies:
    try:
        client.table("scrape_jobs").insert({
            "domain": domain,
            "dataset_id": dataset_id,
            "processed": False,
            "status": "pending"
        }).execute()
        print(f"✓ {domain}")
    except Exception as e:
        print(f"✗ {domain}: {e}")

print(f"\nInserted {len(companies)} companies for dataset_id: dataset002")
