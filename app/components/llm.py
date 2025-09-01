from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info("Initializing HuggingFaceEndpoint for conversational use")
        llm_base = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            task="conversational",   # still needed—even if it might not fully avoid the issue
            huggingfacehub_api_token=hf_token,
            temperature=0.3,
            max_new_tokens=256,
            return_full_text=False,
        )
        llm = ChatHuggingFace(llm=llm_base)

        logger.info("LLM loaded successfully via ChatHuggingFace")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load LLM", e)
        logger.error(str(error_message))
        return None
