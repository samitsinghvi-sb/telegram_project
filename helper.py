from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from models import ContextLLMModel
from collections import defaultdict

from fastapi import FastAPI
from sqlalchemy import select
app = FastAPI()
from telegram import telegram_client
from database import db
from constants import TELEGRAM_MESSAGE_ANALYSIS_PROMPT
from models import TelegramMessageModel,ContextLLMModel,MessageSchedulerModel,UserModel,GroupModel
import json,re
load_dotenv()
from enum import Enum

class GroupEnum(str, Enum):
    MBM_ALUMNI = "MBM CSE Alumni Group"
    PAN_IIT_ALUMNI = "PAN IIT Alumni Group"






def parse_llm_response(raw: str) -> dict:
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    
    # Extract first JSON object found
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No valid JSON found in response")

def analyze_messages_with_llm(messages):
    """
    messages: list of dicts -> [{"msg_id": int, "text": str}]
    llm_call_fn: function that takes prompt and returns LLM response (string)
    """
    formatted_msgs = "\n".join(
        [f"{m['msg_id']}: {m['text']}" for m in messages]
    )
    prompt = TELEGRAM_MESSAGE_ANALYSIS_PROMPT.format(formatted_msgs=formatted_msgs)
    response = llm_call_fn(prompt)
    return response




client = OpenAI(
    api_key=os.getenv("SECRET_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def llm_call_fn(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a JSON-only response bot. Always respond with strict valid JSON. No explanation, no markdown, no extra text."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    return parse_llm_response(raw)  # use defensive parser above



def context_scheduler_insertion(grouped,group_id):
    context_objects_list = []
    offset = 0
    total_count = 0
    for user in grouped.keys():
        llm_req_list = []
        for message in grouped[user]:
            if offset!=0 and offset%5==0:
                llm_resp = analyze_messages_with_llm(llm_req_list)
                if not llm_resp['is_project'] and not llm_resp['is_job_application']:
                    offset+=1
                    continue
                llm_req_list = []
                relevant_messages = ','.join(llm_resp['relevant_messages']) if isinstance(llm_resp['relevant_messages'], list) else llm_resp['relevant_messages']
                context_object = ContextLLMModel(user_id=user,group_id=group_id,is_project=llm_resp['is_project'], is_job_application=llm_resp["is_job_application"], relevant_messages=relevant_messages, summary = llm_resp['summary'])
                total_count+=1
                context_objects_list.append(context_object)
            msg_obj = {
                "msg_id": message.message_id,
                "text": message.text
            }
            llm_req_list.append(msg_obj)
            offset+=1
    return context_objects_list,total_count

def get_grouped_messages(group_id):
    scheduler_obj = db.query(MessageSchedulerModel).filter_by(
        scheduler_name="context_fetch", group_id=group_id).first()
    last_offset = scheduler_obj.offset if scheduler_obj else 0
    if not scheduler_obj:
        scheduler_obj = MessageSchedulerModel(
            scheduler_name="context_fetch",
            group_id=group_id,
            offset=0
        )
        db.add(scheduler_obj)
        db.commit()
    result = db.execute(
    select(TelegramMessageModel).where(
        TelegramMessageModel.message_id > last_offset,
        TelegramMessageModel.group_id == group_id
    )
)
    messages = result.scalars().all()
    grouped = defaultdict(list)
    latest_msg_id = 0
    for message in messages:
        latest_msg_id = message.message_id
        grouped[message.user_id].append(message)
    
    return latest_msg_id,grouped

def get_messages_from_telegram(source_entity,group_id):
    scheduler_obj = db.query(MessageSchedulerModel).filter_by(
        scheduler_name="messages_fetch", group_id=group_id).first()
    last_offset = scheduler_obj.offset if scheduler_obj else 0

    if last_offset == 0:
        iterator = telegram_client.iter_messages(source_entity, reverse=True)
    else:
        iterator = telegram_client.iter_messages(
            source_entity,
            reverse=True,
            min_id=last_offset,
        )
    return iterator

def insert_new_user(user,user_ids,user_objects_list):
    user_obj = UserModel(
    username=user.username,
    first_name=user.first_name,
    last_name=user.last_name,
    user_id=int(user.id)
)
    user_ids.append(user.id)
    user_objects_list.append(user_obj)
    return user_ids,user_objects_list

async def iterate_message_to_insert_in_db(messages,message_list,group_id):
    latest_msg_id = 0
    user_objects_list = []
    all_groups = db.query(GroupModel).all()
    group_ids = [group.group_id for group in all_groups]
    user_ids = db.query(UserModel.user_id).all()
    user_ids = [uid[0] for uid in user_ids]
    async for message in messages:
        if message.text is None or not len(message.text) or not message.sender_id:
            continue
        chat = await message.get_chat()
        user = await telegram_client.get_entity(message.sender_id)

        group_exist = group_id in group_ids
        if not group_exist:
            group_obj = GroupModel(
                group_id=group_id,
                group_name=chat.title if chat.title else "Unknown"
            )
            db.add(group_obj)
            db.commit()
            group_ids.append(group_id)
        if message.sender_id not in user_ids:
            user_ids,user_objects_list = insert_new_user(user,user_ids,user_objects_list)
        latest_msg_id = message.id
        message_obj = TelegramMessageModel(
            message_id=int(message.id),
            text=message.text,
            group_id=int(group_id),
            channel_name="telegram",
            user_id=int(message.sender_id)
        )
        message_list.append(message_obj)
    db.bulk_save_objects(user_objects_list)
    db.commit()
    return message_list,latest_msg_id