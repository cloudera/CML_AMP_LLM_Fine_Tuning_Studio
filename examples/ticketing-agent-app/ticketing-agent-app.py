import streamlit as st
import os
import json
import requests

MODEL_SERVICE_URL = "https://modelservice." + os.getenv("CDSW_DOMAIN") + "/model?accessKey=" + os.getenv("TICKETING_MODEL_ACCESS_KEY")

# \n needs to be \\\\n to handle python % templating AND going through cml model service
#    \\n is enough if python % templating is not used
PROMPT = """You are an event ticketing customer LLM chatbot responsible for generating a one-word, snake_case action, based on a customer input. Please provide the action below based on the input from the customer.\\\\n\\\\n### CUSTOMER: %s\\\\n### ACTION:"""


with open('examples/ticketing-agent-app/customer-help-library.json') as f:
    customer_help_dict = json.load(f)

def find_customer_help(classification):
  if classification.strip() in customer_help_dict:
    return customer_help_dict[classification.strip()]
  else:
    return "Unknown customer help category [%s]." % classification

if 'status_class_generating' not in st.session_state:
    st.session_state['status_class_generating'] = False

if 'gen_class' not in st.session_state:
    st.session_state['gen_class'] = ""
    
if 'customer_input' not in st.session_state:
    st.session_state['customer_input'] = ""

if 'Suggested Customer Service Actions' not in st.session_state:
    st.session_state['Suggested Customer Service Actions'] = ""
    
if 'help_text' not in st.session_state:
    st.session_state['help_text']= """
        **Examples**:
        
        I need to cancel my concert tickets
        
        Can I get the status of my ticket refund?
        
        Let me talk to an agent

    """

def call_cml_model(customer_input):
    # NOTE: newlines in customer_input are not handled properly currently, need to escape them as \\n
    ticketing_model_input = PROMPT % customer_input
    print(ticketing_model_input)
    response = requests.post(MODEL_SERVICE_URL,
                      data='{"request":{"prompt":"%s"}}' % ticketing_model_input,
                      headers={'Content-Type': 'application/json', 'Authorization': 'Bearer %s' % os.getenv('CDSW_APIV2_KEY')})
    response_dict = response.json()
    if 'success' in response_dict:
      print(response_dict)
      return response_dict['response'].strip()
    elif 'errors' in response_dict:
      return "ERROR: " + json.dumps(response_dict['errors'])
    else:
      return "ERROR: Check application logs"

def main():
  st.set_page_config(layout="wide")
  st.title("📝 Events Ticketing Support Agent")

  col1, col2= st.columns([1,1])
  with col1:
    class_container = st.container(border=True)
    with class_container:
      st.markdown("### Customer Input")
      st.caption("Customer Support Request Text")
      st.session_state['customer_input'] = st.text_input(
          "",
          placeholder="Insert sample customer input here...",
          disabled=st.session_state['status_class_generating'],
          value=st.session_state['customer_input'],
          label_visibility="collapsed"
      )
      if st.button("Submit",
               key='class_btn',
              disabled= st.session_state['status_class_generating']):
        # Perform inference using the loaded model
        with st.spinner('Generating...'):
          st.session_state['gen_class'] = call_cml_model(st.session_state['customer_input'])
          st.session_state['Suggested Customer Service Actions'] = find_customer_help(st.session_state['gen_class'])
      st.markdown(st.session_state['help_text'])
      
  with col2:
    suggestion_container = st.container(border=True)
    with suggestion_container:
      st.markdown("""
      ### Support Agent Response
      """)
      class_txt = st.container(border=True)
      st.caption("Detected Customer Intent")
      st.code(body=st.session_state['gen_class'])
      st.caption("Suggested Response to Customer")
      suggestion_txt = st.container(border=True)
      suggestion_txt.write(st.session_state['Suggested Customer Service Actions'])

if __name__ == "__main__":
    main()