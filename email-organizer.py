import streamlit as st
import boto3
from botocore.config import Config
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the AWS Bedrock client
def get_bedrock_client():
    config = Config(
        read_timeout=60,
        connect_timeout=60,
        retries={'max_attempts': 10, 'mode': 'adaptive'}
    )
    return boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        config=config
    )

def summarize_emails(email_content):
    client = get_bedrock_client()
    model_id = 'anthropic.claude-v2'  # Make sure this is the correct model ID

    # Construct the prompt according to Bedrock's required format
    summary_prompt = f"""
Human: Given the following emails, generate a delightful inbox overview that users will see when they open their email app that includes:
1. Total number of emails received today and a breakdown of:
   - Important & urgent emails that need immediate attention.
   - Emails awaiting replies.
   - Potential subscription emails for unsubscription suggestions.
   - Emails flagged for long-term storage.
2. Action items needed for:
   - Urgent emails.
   - Replies needed with suggested deadlines.
   - Subscriptions recommended for unsubscription.
   - Informational emails to store for future reference.
3. Trends analysis including the most active conversation, email volume comparison with last week, and most frequently discussed topics.
4. Weekly reminders for managing the inbox efficiently.
5. Add Emojis and make it personalble and human-like.
Email content:
{email_content}

Assistant:
"""

    logging.info("Sending prompt to model: %s", summary_prompt)

    # Prepare the request to AWS Bedrock
    request_body = json.dumps({
        "prompt": summary_prompt,
        "max_tokens_to_sample": 1000,
        "temperature": 0.7,
        "top_p": 0.9,
    })

    logging.info("Raw request body: %s", request_body)

    # Invoke the AWS Bedrock model
    try:
        response = client.invoke_model(
            body=request_body,
            modelId=model_id,
            accept='application/json',
            contentType='application/json'
        )

        # Check if 'body' is in the response and read it
        if 'body' in response:
            raw_response_content = response['body'].read().decode('utf-8')
            logging.info("Raw response content: %s", raw_response_content)

            response_body = json.loads(raw_response_content)
            logging.info("Parsed JSON response body: %s", response_body)

            if 'completion' in response_body:
                transformed_message = response_body['completion'].strip()
                logging.info("Transformed message being returned: %s", transformed_message)
                return transformed_message

    except Exception as e:
        logging.exception("Exception during model invocation or response processing: %s", e)
        return "Error: Could not transform the message."

st.title("Email Organizer Assistant")

# Pre-filled text box with all sample emails
sample_emails = """Your detailed email content here..."""  # Update your sample emails here

email_content = st.text_area("Paste or Edit Emails Here:", value=sample_emails, height=300)

if st.button("Generate Inbox Summary"):
    if email_content:
        summary = summarize_emails(email_content)
        st.success("Inbox Summary generated!")
        st.write("Inbox Summary:", summary)
    else:
        st.error("Please ensure the email content box is not empty before generating the summary.")
