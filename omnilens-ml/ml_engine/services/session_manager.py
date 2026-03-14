import json
import os
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, filename="session_products.json"):
        self.filepath = os.path.join(os.getcwd(), filename)

    def save_products(self, products: list[dict]):
        """Appends new products to the session file, avoiding duplicates by link."""
        existing = self.load_products()
        
        # Simple deduplication by link
        seen_links = {p.get("external_link") or p.get("link") for p in existing}
        
        new_count = 0
        for p in products:
            link = p.get("external_link") or p.get("link")
            if link not in seen_links:
                existing.append(p)
                seen_links.add(link)
                new_count += 1
                
        try:
            with open(self.filepath, "w") as f:
                json.dump(existing, f, indent=2)
            logger.info(f"Saved {new_count} new products to session. Total: {len(existing)}")
        except Exception as e:
            logger.error(f"Failed to save session products: {e}")

    def load_products(self) -> list[dict]:
        """Loads products from the session file."""
        if not os.path.exists(self.filepath):
            return []
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session products: {e}")
            return []

    def get_seen_links(self) -> set[str]:
        """Returns a set of all product links currently in the session."""
        products = self.load_products()
        return {p.get("external_link") or p.get("link") for p in products if p.get("external_link") or p.get("link")}

    def clear(self):
        """Clears the session file."""
        try:
            with open(self.filepath, "w") as f:
                f.write("[]")
            logger.info("Session products cleared.")
        except Exception as e:
            logger.error(f"Failed to clear session products: {e}")

session_manager = SessionManager()
