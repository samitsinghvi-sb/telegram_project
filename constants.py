TELEGRAM_MESSAGE_ANALYSIS_PROMPT = """\
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

IMPORTANT: Return ONLY the raw JSON object. No explanation, no markdown code blocks, no preamble. Start your response with '{{' and end with '}}'.
"""


group_ids = {
      "MBM CSE Alumni Group": -1001318801474,
      "PAN IIT Alumni Group": -1001458595201
}