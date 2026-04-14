from src.pipelines.indexing_pipeline import IndexingPipeline


class SessionIndexService:
    def __init__(self, pipeline: IndexingPipeline | None = None):
        self.pipeline = pipeline or IndexingPipeline()
        self.session_result = None

    def build_session_index(self, folder_path: str):
        """
        Build a temporary in-memory index for one session.
        """
        self.session_result = self.pipeline.run(folder_path)
        return self.session_result

    def get_vector_store(self):
        if self.session_result is None:
            raise ValueError("No session index has been built yet.")
        return self.session_result["vector_store"]