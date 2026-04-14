from src.services.persistent_index_service import PersistentIndexService

folder_path = "data/raw_pdfs"
save_dir = "storage/pharma_index"

service = PersistentIndexService()
result = service.build_and_save(folder_path, save_dir)

print("\nSaved persistent index.", flush=True)
print(result["stats"], flush=True)