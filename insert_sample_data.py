"""Script to insert sample data into Supabase."""
import re
from supabase import create_client
from config import get_settings

def parse_sample_data(file_path: str) -> list[dict]:
    """Parse the sample data file and extract domains."""
    with open(file_path, "r") as f:
        content = f.read()
    
    url_pattern = r'\[https?://([^\]]+)\]'
    urls = re.findall(url_pattern, content)
    
    domains = []
    for url in urls:
        domain = url.replace("www.", "").split("/")[0]
        domains.append(domain)
    
    return domains


def insert_data(dataset_id: str, domains: list[str]):
    """Insert domains into Supabase."""
    settings = get_settings()
    client = create_client(settings.supabase_url, settings.supabase_key)
    
    rows = [{"dataset_id": dataset_id, "domain": domain} for domain in domains]
    
    print(f"Inserting {len(rows)} rows into scrape_jobs...")
    print(f"Dataset ID: {dataset_id}")
    print(f"Domains: {domains}")
    
    response = client.table("scrape_jobs").insert(rows).execute()
    
    print(f"\nInserted {len(response.data)} rows successfully!")
    for row in response.data:
        print(f"  - ID: {row['id']}, Domain: {row['domain']}")
    
    return response.data


if __name__ == "__main__":
    domains = parse_sample_data("sampledata.txt")
    print(f"Found {len(domains)} domains in sample data:")
    for d in domains:
        print(f"  - {d}")
    print()
    
    insert_data("dataset001", domains)
