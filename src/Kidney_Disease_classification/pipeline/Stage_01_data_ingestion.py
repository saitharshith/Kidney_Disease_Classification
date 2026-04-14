from Kidney_Disease_classification.config.configuration import CofigurationManager
from Kidney_Disease_classification.components.Data_ingestion import DataIngestion
from Kidney_Disease_classification import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass 
    
    def main(self):
        config = CofigurationManager()
        data_ingestion_config = config.get_data_ingestion_cofig()
        data_ingestion = DataIngestion(config = data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == "__main__":
    try:
        logger.info(f">>>Stage {STAGE_NAME} Started <<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>> Stage {STAGE_NAME} Completed<<<")
    except Exception as e:
        logger.exception(e)
        raise e 