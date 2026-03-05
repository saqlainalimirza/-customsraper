"""Reset failed rows to try again."""
from supabase import create_client
from config import get_settings

settings = get_settings()
client = create_client(settings.supabase_url, settings.supabase_key)

response = client.table("scrape_jobs").update({
    "processed": False,
    "status": "pending",
    "error_message": None
}).eq("dataset_id", "dataset001").execute()

print(f"Reset {len(response.data)} rows")
for row in response.data:
    print(f"  - {row['domain']}: status={row['status']}, processed={row['processed']}")
