from collections import defaultdict

from fastapi import FastAPI
from sqlalchemy import select
app = FastAPI()
from telegram import client
from database import db
from datetime import datetime
from models import TelegramMessageModel, UserModel, MessageSchedulerModel,ContextLLMModel
from helper import analyze_messages_with_llm


@app.post("/send_otp")
async def send_otp(phone_number: str):
    await client.connect()
    await client.send_code_request(phone_number)
    return {"message": f"OTP sent to {phone_number}"}

@app.post("/login")
async def login(phone_number: str, otp: str):
    await client.sign_in(phone_number, otp)
    return {"message": f"Logged in as {phone_number}"}

@app.get('/fetch_messages')
async def fetch_messages():
    await client.connect()
    source_entity = await client.get_entity(-1003588865018)

    message_list = []
    latest_msg_id = 0
    last_offset = db.query(MessageSchedulerModel).filter_by(
        scheduler_name="messages_fetch").first().offset
    if last_offset == 0:
        iterator = client.iter_messages(source_entity, reverse=True)
    else:
        iterator = client.iter_messages(
            source_entity,
            reverse=True,
            min_id=last_offset
        )

    user_ids = db.query(UserModel.user_id).all()
    user_ids = [uid[0] for uid in user_ids]
    async for message in iterator:
        if message.text is None or not len(message.text):
            continue
        chat = await message.get_chat()
        user = await client.get_entity(message.sender_id)

        if message.sender_id not in user_ids:
            user_obj = UserModel(
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
                user_id=int(user.id)
            )
            db.add(user_obj)
            user_ids.append(user.id)
            db.commit()
        db.rollback()
        latest_msg_id = message.id
        message_obj = TelegramMessageModel(
            message_id=int(message.id),
            text=message.text,
            group_id=int(chat.id),
            channel_name="telegram",
            group_name=chat.title,
            user_id=int(message.sender_id)
        )
        message_list.append(message_obj)
    

    scheduler_obj = db.query(MessageSchedulerModel).filter_by(scheduler_name="messages_fetch").first()
    scheduler_obj.last_scheduler_date = datetime.now()
    scheduler_obj.offset = latest_msg_id
    db.bulk_save_objects(message_list)
    db.commit()


@app.get('/context_scheduler')
async def context_scheduler():
    result = db.execute(select(TelegramMessageModel))
    messages = result.scalars().all()

    # Group by user_id
    grouped = defaultdict(list)
    for message in messages:
        grouped[message.user_id].append(message)
    
    
    final_llm_responses = {}
    for user in grouped.keys():
        llm_req_list = []
        group_id = None
        for message in grouped[user]:
            group_id = message.group_id
            msg_obj = {
                "msg_id": message.message_id,
                "text": message.text
            }
            llm_req_list.append(msg_obj)
        llm_resp = analyze_messages_with_llm(llm_req_list)
        if not llm_resp['is_project'] and not llm_resp['is_job_application']:
            continue
        relevant_messages = ','.join(llm_resp['relevant_messages']) if isinstance(llm_resp['relevant_messages'], list) else llm_resp['relevant_messages']
        context_object = ContextLLMModel(user_id=user,group_id=group_id,is_project=llm_resp['is_project'], is_job_application=llm_resp["is_job_application"], relevant_messages=relevant_messages, summary = llm_resp['summary'])
        db.add(context_object)
        db.commit()
        final_llm_responses[user] = llm_resp
    
    return {"message": final_llm_responses}


@app.get('/fetch_chats')
async def get_all_chats():
    chats = []

    async for dialog in client.iter_dialogs():
        chats.append({
            "name": dialog.name,
            "id": dialog.id,
            "is_group": dialog.is_group,
            "is_channel": dialog.is_channel
        })

    return chats


@app.get('/disconnect')
async def disconnect():
    await client.disconnect()
    return {"message": "Disconnected from Telegram"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
