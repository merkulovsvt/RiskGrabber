import warnings

# Отключить предупреждения Pydantic о сериализации (parsed/structured output в LangChain/OpenAI)
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
)
