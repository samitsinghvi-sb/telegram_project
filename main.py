from collections import defaultdict

from fastapi import FastAPI
from sqlalchemy import select
app = FastAPI()
from telegram import telegram_client
from database import db
from datetime import datetime
from models import TelegramMessageModel, UserModel, MessageSchedulerModel,ContextLLMModel
from helper import analyze_messages_with_llm,context_scheduler_insertion,get_grouped_messages,get_messages_from_telegram,insert_new_user,iterate_message_to_insert_in_db


@app.post("/send_otp")
async def send_otp(phone_number: str):
    await telegram_client.connect()
    await telegram_client.send_code_request(phone_number)
    return {"message": f"OTP sent to {phone_number}"}

@app.post("/login")
async def login(phone_number: str, otp: str):
    await telegram_client.sign_in(phone_number, otp)
    return {"message": f"Logged in as {phone_number}"}

@app.get('/fetch_messages')
async def fetch_messages():
    await telegram_client.connect()
    #the below line is used to connect to a particular group so that we can fetch messages from it
    source_entity = await telegram_client.get_entity(-1003588865018)
    message_list = []
    latest_msg_id = 0
    messages = get_messages_from_telegram(source_entity)
    user_ids = db.query(UserModel.user_id).all()
    user_ids = [uid[0] for uid in user_ids]
    message_list,latest_msg_id = await iterate_message_to_insert_in_db(message_list=message_list,messages=messages,user_ids=user_ids)
    scheduler_obj = db.query(MessageSchedulerModel).filter_by(scheduler_name="messages_fetch").first()
    scheduler_obj.last_scheduler_date = datetime.now()
    scheduler_obj.offset = latest_msg_id
    db.bulk_save_objects(message_list)
    db.commit()


@app.get('/context_scheduler')
async def context_scheduler():
    latest_msg_id,grouped_messages = get_grouped_messages()
    context_objects_list,total_count = context_scheduler_insertion(grouped_messages)
    scheduler_obj = db.query(MessageSchedulerModel).filter_by(scheduler_name="context_fetch").first()
    scheduler_obj.last_scheduler_date = datetime.now()
    scheduler_obj.offset = latest_msg_id
    db.add(scheduler_obj)
    db.bulk_save_objects(context_objects_list)
    db.commit()
    
    return {"message": "Contexts added successfully : {}".format(total_count)}


@app.get('/fetch_chats')
async def get_all_chats():
    chats = []

    async for dialog in telegram_client.iter_dialogs():
        chats.append({
            "name": dialog.name,
            "id": dialog.id,
            "is_group": dialog.is_group,
            "is_channel": dialog.is_channel
        })

    return chats


@app.get('/disconnect')
async def disconnect():
    await telegram_client.disconnect()
    return {"message": "Disconnected from Telegram"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
