import csv
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

client = create_client(url, key)

# Extract domain from website URL
def extract_domain(website_url):
    if not website_url or website_url.strip() == '':
        return None
    url = website_url.strip().lower()
    # Remove https:// or http://
    if url.startswith('https://'):
        url = url[8:]
    elif url.startswith('http://'):
        url = url[7:]
    # Remove trailing slash
    url = url.rstrip('/')
    return url

# Read CSV and insert
success = 0
failed = 0
duplicates = 0

with open('cleaned_data.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i % 100 == 0:
            print(f"Processing row {i}...")
        
        domain = extract_domain(row.get('Website', ''))
        
        if not domain:
            failed += 1
            continue
        
        try:
            client.table("scrape_jobs").insert({
                "domain": domain,
                "dataset_id": "dataset003",
                "processed": False,
                "status": "pending"
            }).execute()
            success += 1
        except Exception as e:
            if "duplicate" in str(e).lower():
                duplicates += 1
            else:
                failed += 1

print(f"\n✓ Inserted: {success}")
print(f"✗ Failed: {failed}")
print(f"⚠ Duplicates: {duplicates}")
print(f"Total: {success + failed + duplicates}")
