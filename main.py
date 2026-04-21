import asyncio
from fastapi import FastAPI
from telegram import telegram_client
from database import db
from datetime import datetime
from models import TelegramMessageModel, UserModel, MessageSchedulerModel
from helper import context_scheduler_insertion,get_grouped_messages,iterate_message_to_insert_in_db,GroupEnum,scheduler_updation_with_latest_offset
from constants import group_ids

app = FastAPI()

otp_store = {}

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
    """Fetch messages from a Telegram group and store them in the database."""
    await telegram_client.connect()
    group_id = group_ids.get(group_name.value)
    if not group_id:
        return {"error": "Invalid group name"}
    message_list,latest_msg_id,user_objects_list = await iterate_message_to_insert_in_db(group_id=group_id)
    scheduler_updation_with_latest_offset(group_id,latest_msg_id)
    db.bulk_save_objects(message_list)
    db.bulk_save_objects(user_objects_list)
    db.commit()
    return {"message": f"Fetched {len(message_list)} messages from {group_name.value} group"}


@app.get('/context_scheduler')
async def context_scheduler(group_name : GroupEnum):
    """"API to fetch messages from database, analyze with LLM and store the relevant contexts in another table."""
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
