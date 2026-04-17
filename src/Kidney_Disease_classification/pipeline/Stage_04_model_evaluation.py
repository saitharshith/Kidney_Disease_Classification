from Kidney_Disease_classification.config.configuration import CofigurationManager
from Kidney_Disease_classification.components.Model_Evaluation import Evaluation
from Kidney_Disease_classification import logger


STAGE_NAME = "Model Evaluation"
class ModelEvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = CofigurationManager()
        eval_config = config.get_eval_config()
        evaluation = Evaluation(config=eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
