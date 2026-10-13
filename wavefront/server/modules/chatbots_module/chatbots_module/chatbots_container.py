from dependency_injector import containers, providers

from db_repo_module.models.chat_message import ChatMessage
from db_repo_module.models.chat_session import ChatSession
from db_repo_module.models.chatbot import Chatbot
from db_repo_module.models.namespace import Namespace
from db_repo_module.repositories.sql_alchemy_repository import SQLAlchemyRepository

from chatbots_module.services.chat_inference_service import ChatInferenceService
from chatbots_module.services.chat_session_service import ChatSessionService
from chatbots_module.services.chatbot_service import ChatbotService
from chatbots_module.services.voice_session_service import VoiceSessionService


class ChatbotsContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    # External dependencies
    db_client = providers.Dependency()
    cache_manager = providers.Dependency()
    llm_inference_config_service = providers.Dependency()
    # Optional: chat runs unguarded when no engine is supplied, so this
    # container stays usable from entry points that do not build the
    # guardrails stack.
    guardrails_engine = providers.Dependency(default=None)
    # Optional: voice sessions resolve their tools through these; without them
    # a voice chatbot can only be configured without tools.
    tool_loader = providers.Dependency(default=None)
    api_services_manager = providers.Dependency(default=None)

    # Repositories
    chatbot_repository = providers.Singleton(
        SQLAlchemyRepository[Chatbot],
        model=Chatbot,
        db_client=db_client,
    )

    chat_session_repository = providers.Singleton(
        SQLAlchemyRepository[ChatSession],
        model=ChatSession,
        db_client=db_client,
    )

    chat_message_repository = providers.Singleton(
        SQLAlchemyRepository[ChatMessage],
        model=ChatMessage,
        db_client=db_client,
    )

    namespace_repository = providers.Singleton(
        SQLAlchemyRepository[Namespace],
        model=Namespace,
        db_client=db_client,
    )

    # Services
    chatbot_service = providers.Singleton(
        ChatbotService,
        chatbot_repository=chatbot_repository,
        namespace_repository=namespace_repository,
        llm_inference_config_service=llm_inference_config_service,
    )

    chat_session_service = providers.Singleton(
        ChatSessionService,
        chat_session_repository=chat_session_repository,
        chat_message_repository=chat_message_repository,
    )

    chat_inference_service = providers.Singleton(
        ChatInferenceService,
        llm_inference_config_service=llm_inference_config_service,
        guardrails_engine=guardrails_engine,
        azure_openai_api_version=config.model.azure_openai_api_version,
    )

    voice_session_service = providers.Singleton(
        VoiceSessionService,
        llm_inference_config_service=llm_inference_config_service,
        chat_session_service=chat_session_service,
        tool_loader=tool_loader,
        api_services_manager=api_services_manager,
    )
