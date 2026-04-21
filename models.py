from sqlalchemy import Boolean, Column, DateTime, Integer, String, func, BigInteger
from database import Base


class BaseModel():
    created_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now()
    )

    updated_at = Column(
        DateTime,
        nullable=True,
        onupdate=func.now()
    )

class UserModel(BaseModel,Base):
    __tablename__ = "sb_telegram_users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), nullable=True)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    user_id = Column(BigInteger, unique=True, index=True)

class GroupModel(BaseModel, Base):
    __tablename__ = "sb_telegram_groups"

    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(BigInteger, unique=True)
    group_name = Column(String(500))


class TelegramMessageModel(BaseModel,Base):
    __tablename__ = "sb_telegram_messages"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, unique=True, index=True)
    user_id = Column(BigInteger)
    group_id = Column(BigInteger)
    channel_name = Column(String(100))
    text = Column(String(1000))

class MessageSchedulerModel(Base):
    __tablename__ = "sb_scheduler_master"

    id = Column(Integer, primary_key=True, index=True)
    last_scheduler_date = Column(DateTime)
    offset = Column(Integer)
    scheduler_name = Column(String(100), unique=True, index=True)
    group_id = Column(BigInteger)

class ContextLLMModel(BaseModel,Base):
    __tablename__ = "sb_context_llm"

    id = Column(Integer, primary_key=True, index=True)
    relevant_messages = Column(String, unique=True, index=True)
    summary = Column(String(1000))
    user_id = Column(BigInteger)
    group_id = Column(BigInteger)
    is_project = Column(Boolean, default=False)
    is_job_application = Column(Boolean, default=False)
