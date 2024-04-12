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

def summarize_emails(email_contents):
    client = get_bedrock_client()
    model_id = 'anthropic.claude-v2'  # Adjust if you have a specific model version

    # Format the email contents into a single string with clear delineation
    formatted_emails = "\n\n".join([f"Email {idx + 1}:\n{content}" for idx, content in enumerate(email_contents)])
    
    # Construct the prompt with detailed instructions for the desired output format
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
{formatted_emails}

Assistant:
"""

    logging.info("Sending prompt to model: %s", summary_prompt)

    # Configure the request to AWS Bedrock
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

        # Decode the response
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

# Create three text areas for inputting emails with placeholder text for a demo
email_contents = []
for i in range(1, 4):  # Creating three input boxes
    email = st.text_area(f"Email {i}", value=f"Placeholder for email {i}. Please replace with your own email content.", height=150)
    email_contents.append(email)

if st.button("Generate Inbox Summary"):
    if all(email_contents):
        summary = summarize_emails(email_contents)
        st.success("Inbox Summary generated!")
        st.write("Inbox Summary:", summary)
    else:
        st.error("Please ensure all email boxes are filled before generating the summary.")
