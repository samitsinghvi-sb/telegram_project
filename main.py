import asyncio
from collections import defaultdict

from fastapi import FastAPI
from sqlalchemy import select
app = FastAPI()
from telegram import telegram_client
from database import db
from datetime import datetime
from models import TelegramMessageModel, UserModel, MessageSchedulerModel,ContextLLMModel
from helper import analyze_messages_with_llm,context_scheduler_insertion,get_grouped_messages,get_messages_from_telegram,insert_new_user,iterate_message_to_insert_in_db,GroupEnum
from constants import group_ids

otp_store = {}


@app.on_event("startup")
async def startup():
    await telegram_client.connect()


@app.post("/send_otp")
async def send_otp(phone_number: str):
    if not telegram_client.is_connected():
        await telegram_client.connect()
    await asyncio.sleep(0.5)
    result = await telegram_client.send_code_request(phone_number)

    # store phone_code_hash
    otp_store[phone_number] = result.phone_code_hash
    return {"message": f"OTP sent to {phone_number}"}


@app.post("/login")
async def login(phone_number: str, otp: str):
    phone_code_hash = otp_store.get(phone_number)

    if not phone_code_hash:
        raise Exception("OTP not requested or expired")

    await telegram_client.sign_in(
        phone=phone_number,
        code=otp,
        phone_code_hash=phone_code_hash
    )

    return {"message": f"Logged in as {phone_number}"}


@app.get('/fetch_messages')
async def fetch_messages(group_name : GroupEnum ):
    await telegram_client.connect()
    #the below line is used to connect to a particular group so that we can fetch messages from it
    group_id = group_ids.get(group_name.value)
    if not group_id:
        return {"error": "Invalid group name"}
    source_entity = await telegram_client.get_entity(group_id)
    message_list = []
    latest_msg_id = 0
    messages = get_messages_from_telegram(source_entity,group_id)
    message_list,latest_msg_id = await iterate_message_to_insert_in_db(message_list=message_list,messages=messages,group_id=group_id)
    scheduler_obj = db.query(MessageSchedulerModel).filter_by(scheduler_name="messages_fetch", group_id=group_id).first()
    if not scheduler_obj:
        scheduler_obj = MessageSchedulerModel(
            scheduler_name="messages_fetch",
            group_id=group_id,
        )
        db.add(scheduler_obj)
    scheduler_obj.last_scheduler_date = datetime.now()
    scheduler_obj.offset = latest_msg_id
    db.bulk_save_objects(message_list)
    db.commit()
    return {"message": f"Fetched {len(message_list)} messages from {group_name.value} group"}


@app.get('/context_scheduler')
async def context_scheduler(group_name : GroupEnum):
    group_id = group_ids.get(group_name.value)
    latest_msg_id,grouped_messages = get_grouped_messages(group_id)
    context_objects_list,total_count = context_scheduler_insertion(grouped_messages,group_id)
    scheduler_obj = db.query(MessageSchedulerModel).filter_by(scheduler_name="context_fetch", group_id=group_id).first()
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

@app.get('/get_message')
def get_message(message_id: int, group_id: int):
    message = db.query(TelegramMessageModel).filter_by(message_id=message_id, group_id=group_id).first()
    if not message:
        return {"error": "Message not found"}
    return {
        "message_id": message.message_id,
        "text": message.text,
        "user_id": message.user_id,
        "group_id": message.group_id,
        "channel_name": message.channel_name,
        "group_name": message.group_name
    }


@app.get('/disconnect')
async def disconnect():
    await telegram_client.disconnect()
    return {"message": "Disconnected from Telegram"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
