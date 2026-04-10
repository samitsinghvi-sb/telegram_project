from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()
import json, re

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
    # Step 1: format messages
    formatted_msgs = "\n".join(
        [f"{m['msg_id']}: {m['text']}" for m in messages[5:]]
    )

    # Step 2: prompt
    prompt = f"""
You are a strict JSON generator.

Analyze the messages and determine:

1. Whether ANY subset forms:
   - PROJECT REQUIREMENT
   - JOB APPLICATION

2. If found:
   - Store all msg ids(it must be string) which had relevance in a list in the variable relevant_messages
   - Generate a short summary

3. If none found:
   - relevant_messages = null

Definitions:
- PROJECT REQUIREMENT: building apps, features, tech work
- JOB APPLICATION: resume, hiring, jobs, engineers needed

Output format:
{{
  "relevant_messages": list or empty list,
  "summary": "string",
  "is_project": true or false,
  "is_job_application": true or false
}}

Rules:
- Output ONLY JSON
- Use double quotes
- No extra text

Messages:
{formatted_msgs}

IMPORTANT: Return ONLY the raw JSON object. No explanation, no markdown code blocks, no preamble. Start your response with '{' and end with '}'.

"""



    # Step 3: call LLM
    response = llm_call_fn(prompt)

    # Step 4: parse JSON safely
    return response




client = OpenAI(
    api_key=os.getenv("SECRET_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

import json

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

