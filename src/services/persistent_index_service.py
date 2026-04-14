from src.pipelines.indexing_pipeline import IndexingPipeline


class PersistentIndexService:
    def __init__(self, pipeline: IndexingPipeline | None = None):
        self.pipeline = pipeline or IndexingPipeline()

    def build_and_save(self, folder_path: str, save_dir: str):
        """
        Build an index from documents and persist it to disk.
        """
        result = self.pipeline.run(folder_path)
        result["vector_store"].save(save_dir)
        return result