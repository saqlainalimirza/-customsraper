from typing import Any
from supabase import create_client, Client
from datetime import datetime, timezone

from config import get_settings
from utils.logging import setup_logger

logger = setup_logger(__name__)


class SupabaseClient:
    TABLE_NAME = "scrape_jobs"

    def __init__(self):
        settings = get_settings()
        self.client: Client = create_client(settings.supabase_url, settings.supabase_key)

    async def get_unprocessed(self, dataset_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch unprocessed rows for a given dataset_id."""
        logger.info(f"Fetching up to {limit} unprocessed rows for dataset_id={dataset_id}")
        
        response = (
            self.client.table(self.TABLE_NAME)
            .select("*")
            .eq("dataset_id", dataset_id)
            .eq("processed", False)
            .limit(limit)
            .execute()
        )
        
        rows = response.data or []
        logger.info(f"Found {len(rows)} unprocessed rows")
        return rows

    async def get_failed(self, dataset_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch failed rows for a given dataset_id."""
        logger.info(f"Fetching up to {limit} failed rows for dataset_id={dataset_id}")

        response = (
            self.client.table(self.TABLE_NAME)
            .select("*")
            .eq("dataset_id", dataset_id)
            .eq("status", "failed")
            .limit(limit)
            .execute()
        )

        rows = response.data or []
        logger.info(f"Found {len(rows)} failed rows")
        return rows

    async def update_row(self, row_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Update a row with results and mark as processed."""
        logger.info(f"Updating row {row_id}")
        
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        response = (
            self.client.table(self.TABLE_NAME)
            .update(data)
            .eq("id", row_id)
            .execute()
        )
        
        if response.data:
            logger.info(f"Successfully updated row {row_id}")
            return response.data[0]
        
        logger.warning(f"No data returned after updating row {row_id}")
        return {}

    async def update_status(self, row_id: str, status: str, error_message: str | None = None) -> None:
        """Update the status of a row."""
        data = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if error_message:
            data["error_message"] = error_message
        
        logger.info(f"Updating row {row_id} status to '{status}'")
        
        self.client.table(self.TABLE_NAME).update(data).eq("id", row_id).execute()

    async def mark_completed(
        self,
        row_id: str,
        all_urls: list[str],
        filtered_urls: list[str],
        scraped_content: dict[str, str],
        extracted_answer: str,
        filter_input_tokens: int,
        filter_output_tokens: int,
        extract_input_tokens: int,
        extract_output_tokens: int,
    ) -> dict[str, Any]:
        """Mark a row as completed with all results and token counts."""
        total_tokens = (
            filter_input_tokens + filter_output_tokens +
            extract_input_tokens + extract_output_tokens
        )
        
        data = {
            "processed": True,
            "status": "completed",
            "error_message": None,
            "all_urls": all_urls,
            "filtered_urls": filtered_urls,
            "scraped_content": scraped_content,
            "extracted_answer": extracted_answer,
            "filter_input_tokens": filter_input_tokens,
            "filter_output_tokens": filter_output_tokens,
            "extract_input_tokens": extract_input_tokens,
            "extract_output_tokens": extract_output_tokens,
            "total_tokens": total_tokens,
        }
        
        logger.info(
            f"Marking row {row_id} as completed | "
            f"Total tokens: {total_tokens} (filter: {filter_input_tokens}+{filter_output_tokens}, "
            f"extract: {extract_input_tokens}+{extract_output_tokens})"
        )
        
        return await self.update_row(row_id, data)

    async def mark_failed(self, row_id: str, error_message: str) -> None:
        """Mark a row as failed with an error message."""
        logger.error(f"Marking row {row_id} as failed: {error_message}")
        await self.update_status(row_id, "failed", error_message)
        await self.update_row(row_id, {"processed": True})
