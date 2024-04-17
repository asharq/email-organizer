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

    # Format the email contents into a single string with clear delineation using explicit separators
    formatted_emails = "\n\n--- Email Start ---\n".join([f"Email {idx + 1}:\n{content}" for idx, content in enumerate(email_contents)])
    formatted_emails = f"{formatted_emails}\n--- Email End ---"

    # Construct the prompt with detailed instructions and clear formatting for the desired output
    summary_prompt = f"""
Human: Please generate a delightful inbox overview that includes:
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
5. Add Emojis and make it personable and human-like.

**Emails Content**:
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

def label_emails(email_contents):
    client = get_bedrock_client()
    model_id = 'anthropic.claude-v2'  # Adjust if you have a specific model version

    # Construct the labeling prompt with explicit instruction to suggest folders
    labeling_prompt = "Human:\n"
    labeling_prompt += "\n\n".join([
        f"Email {idx + 1} Content:\n{content}\n\nBased on the content, which folder would you categorize this email into? Suggest one of the typical email folders like Travel, Receipts, Work, Personal, Subscriptions, or Others."
        for idx, content in enumerate(email_contents)
    ])
    labeling_prompt += "\nAssistant:"

    request_body = json.dumps({
        "prompt": labeling_prompt,
        "max_tokens_to_sample": 150,
        "temperature": 0.5,
        "top_p": 0.9,
    })

    logging.info("Sending labeling prompt to model: %s", labeling_prompt)

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
            logging.info("Raw response content from labeling: %s", raw_response_content)

            response_body = json.loads(raw_response_content)
            logging.info("Parsed JSON response body from labeling: %s", response_body)

            if 'completion' in response_body:
                labels = response_body['completion'].split('\n')
                return labels

    except Exception as e:
        logging.exception("Exception during model invocation or response processing for labeling: %s", e)
        return ["Error in labeling each email"]



st.title("Email Organizer Assistant")

# Create three text areas for inputting emails with preset emails for a demo
preset_emails = [
    "Subject: Meeting Confirmation\nDear Jane,\n\nI've scheduled our project meeting for next Monday at 10 AM. Please confirm your attendance and let me know if you have any items to add to the agenda. Looking forward to your insights!\n\nBest,\nTom",
    "Subject: Subscription Renewal Reminder\n\nDear Subscriber,\n\nYour annual subscription for 'Tech Trends' is due to expire next week. Take advantage of our early renewal discount by renewing today. Don't miss out on another year of cutting-edge insights!\n\nBest Regards,\nThe 'Tech Trends' Team",
    "Subject: Request for Document Submission\n\nHi,\n\nCould you please send over the documents we discussed in our last call? We need them by end of this week for the review process. It's important that we stick to the timeline to ensure everything is processed on time.\n\nThanks,\nLisa"
]
# Create three text areas for inputting emails with preset emails for a demo
email_contents = []
for i in range(1, 4):  # Creating three input boxes
    email = st.text_area(f"Email {i}", value=preset_emails[i-1], height=150)
    email_contents.append(email)

col1, col2 = st.columns(2)
with col1:
    if st.button("Generate Inbox Summary"):
        if all(email_contents):
            summary = summarize_emails(email_contents)
            st.success("Inbox Summary generated!")
            st.write("Inbox Summary:", summary)
        else:
            st.error("Please ensure all email boxes are filled before generating the summary.")

with col2:
    if st.button("Organize Emails"):
        if all(email_contents):
            labels = label_emails(email_contents)
            st.success("Emails have been organized!")
            for idx, label in enumerate(labels, start=1):
                st.write(f"Email {idx} Label: {label}")
        else:
            st.error("Please ensure all email boxes are filled before generating the labels.")
