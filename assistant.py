from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

import time
import json
client = OpenAI(api_key="ENTER YOUR API KEY")

# Langchian Document Loader
loader = PyPDFLoader("Enter Your Pdf Path")   #For Pdf Document
# loader = UnstructuredWordDocumentLoader("Enter Your Doc/Docx Path") #For Doc and Docx Documnet

datall = loader.load()
Input = ''.join([page.page_content for page in datall])
#print(len(Input))


# Assiatnt Instrction

Instruction = """
    You are an expert AI assistant responsible for parsing resumes. Your task is to extract and structure resume data into a specific JSON format, adhering to the provided schema with precision. Your primary objective is to ensure accuracy, adherence to the schema, and calling appropriate tool functions for each step.

    ### KEEP THE FOLLOWING POINTS IN MIND:

    - Strictly follow the schema provided for data extraction. If a field is missing in the resume, use the schema’s default values (e.g., null, "", or empty arrays).

    - Ensure all information extracted matches the resume's content. Fix formatting issues, merge split words, and handle any unusual characters while preserving the original meaning.

    - Remove non-standard characters (like bullet points, em dashes, curly quotes, and special symbols). Replace problematic characters with standard ones and correct word splits.

    - Maintain proper capitalization for names and titles. Correct spelling errors, but avoid changing the original language or meaning.

    - Extract only the year for 'Start_Year' and 'End_Year' from date ranges (e.g., 'Jan 2020 - Feb 2023' should be extracted as '{\'Start_Year\': \"2020\", \"End_Year\": \"2023\"}').\n- **Employment_History:** Extract both the month and year. Convert numeric dates (e.g., '01/2022') to 'MMM YYYY' format (e.g., 'Jan 2022'). If a date range ends with 'to-date', 'current', 'Present', or similar, use 'Present' for the 'End_Year' or 'End_Year_month'. If no month is available, list year only.

    - After extraction, ensure all text is clean, no special characters remain, and correct any split words or formatting errors.

    - Extract action-based achievements and responsibilities from the resume and categorize them accurately in the correct sections.

    - Do not omit any information. Ensure all content from the resume is categorized correctly into the appropriate JSON fields.

    ### INSTRUCTIONS:

    - Your Task is to follow these these steps in sequence to extract the required data:

    STEP 1: **Call Employment_History function:**

    STEP 2: **Call Personal_Information function:**

    STEP 3: **Call Additional_Information function:**

    STEP 4: **Call Memberships function:**

    STEP 5: **Call Certifications_courses function:**

    STEP 6: **Call Employement_Summary function:**


### IT IS VERY CRUCIAL THAT YOU CALL ALL THE FUCTIONS IN THE GIVEN SEQUENCE IN ORDER TO EXTRACT THE CORRECT INFORMATION.

"""

## Assiatnt Setup

def chatgpt_process(Input, Instruction):
    print(len(Input))

    assistant = client.beta.assistants.create(
        name="Data Extract",
        instructions=Instruction,
        description="Your main task is to **call all tools strictly** and extract the required data.",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Employment_History",
                    "description": "Extract the employment history from the provided input based on the JSON schema.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Input containing employment details for extraction."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Additional_information",
                    "description": "Extract additional information such as interests, skills, etc., based on the JSON schema.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Input containing additional information like interests and skills."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Memberships",
                    "description": "Extract membership details from the provided input based on the JSON schema.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Input containing membership details for extraction."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Certifications_courses",
                    "description": "Extract certifications and course details from the provided input based on the JSON schema.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Input containing certification and course details for extraction."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Employement_Summary",
                    "description": "Extract the Employement summary from the provided input based on the JSON schema.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Input containing a summary of the Employment for extraction."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Personal_Information",
                    "description": "Extract personal information from the provided input based on the JSON schema.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Input containing personal information details for extraction."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
        ],
        model="gpt-4o",
    )
    return assistant


assistant_id = chatgpt_process(Input,Instruction).id
thread = client.beta.threads.create()
thread = thread.id



def continuous_chat(client, assistant_id, i, thread):
    user_input = i
    client.beta.threads.messages.create(thread, role="user", content=str(user_input))
    messages = run_assistant(client, assistant_id, thread,user_input)
    if messages:
        message_dict = json.loads(messages.model_dump_json())
        assistant_response = message_dict['data'][0]['content'][0]["text"]["value"]
        return assistant_response
    else:
        return "No messages received from the assistant."


def run_assistant(client, assistant_id, thread_id,user_input):
    # Create a new run for the given thread and assistant
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    while run.status in ["in_progress", "queued"]:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        print("run.status : ",run.status)
        if run.status == "completed":
            return client.beta.threads.messages.list(thread_id=thread_id)
        if run.status == "requires_action":
            tool_outputs = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
              print(tool_call.function.name,"================tool_call.function.name====================")
              if tool_call.function.name == "Employment_History":
                  img = Employment_History(user_input)
                  tool_outputs.append({
                      "tool_call_id": tool_call.id,
                      "output": img
                  })
              elif tool_call.function.name == "Additional_information":
                  img = Additional_information(user_input)
                  tool_outputs.append({
                      "tool_call_id": tool_call.id,
                      "output": img
                  })
              elif tool_call.function.name == "Memberships":
                  img = Memberships(user_input)
                  tool_outputs.append({
                      "tool_call_id": tool_call.id,
                      "output": img
                  })
              elif tool_call.function.name == "Certifications_courses":
                  img = Certifications_courses(user_input)
                  tool_outputs.append({
                      "tool_call_id": tool_call.id,
                      "output": img
                  })
              elif tool_call.function.name == "Employement_Summary":
                  img = Employement_Summary(user_input)
                  tool_outputs.append({
                      "tool_call_id": tool_call.id,
                      "output": img
                  })
              elif tool_call.function.name == "Personal_Information":
                  img = Personal_Information(user_input)
                  tool_outputs.append({
                      "tool_call_id": tool_call.id,
                      "output": img
                  })
            if tool_outputs:
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )

def generate_response(Input, system_message):
    response = client.chat.completions.create(
        model="gpt-4o",  # Assuming you meant "gpt-4"
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": Input}
        ],
        temperature=0.1,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=16380,
    )
    return response.choices[0].message.content


data = []

def Employment_History(Input):
    system_message = """
        You are tasked with analyzing the provided Input and extracting relevant text values for specified entities.
        Follow the given JSON schema accurately to ensure all details are correctly captured don't skip any Information.
        If a field is not present, use the default value empty string.

        Output: Employment_History
        The output should strictly adhere to the following JSON schema:
        {
          "Employment_History": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "Company_name": { "type": "string" },
                "Start_Month_Year": { "type": "string" },
                "End_Month_Year": { "type": "string"},
                "Job_title": { "type": "string" },
                "Location": { "type": "string" },
                "Description": { "type": "string" },
                "Key_Responsibilities_and_Achievements": {
                  "type": "array",
                  "items": { "type": "string" }
                }
              }
            },
            "required": ["Company_name", "Start_Month_Year", "End_Month_Year", "Job_title"]
          }
        }
    ### **ENSURE THAT EVERY EMPLOYMENT HISTORY IS LISTER IN THE OUTPUT, THIS IS VERY CRUCIAL STEP FOR THE OUTPUT RESPONSE **
    """
    result = generate_response(Input, system_message)
    cleaned_response_text = result.replace('```json', '').replace('```', '').strip()
    print("Cleaned response text:")
    print(cleaned_response_text)
    data.append(cleaned_response_text)
    return result


def Additional_information(Input):
    system_message = """
        You are tasked with analyzing the provided Input and extracting relevant text values for specified entities.
        Follow the given JSON schema accurately to ensure all details are correctly captured don't skip any Information.
        If a field is not present, use the default value empty string.

        Output: Additional_information
        The output should strictly adhere to the following JSON schema:
        {
          "Additional_information": {
            "type": "object",
            "properties": {
              "Skills": { "type": "array", "items": { "type": "string", "default": "" } },
              "Training_and_awards": { "type": "array", "items": { "type": "string", "default": "" } },
              "Interests": { "type": "array", "items": { "type": "string", "default": "" } },
              "Languages": { "type": "array", "items": { "type": "string", "default": "" } }
            }
          }
        }
    """
    result = generate_response(Input, system_message)
    cleaned_response_text = result.replace('```json', '').replace('```', '').strip()
    print("Cleaned response text:")
    print(cleaned_response_text)
    data.append(cleaned_response_text)
    return result


def Memberships(Input):
    system_message = """
        You are tasked with analyzing the provided Input and extracting relevant text values for specified entities.
        Follow the given JSON schema accurately to ensure all details are correctly captured don't skip any Information.
        If a field is not present, use the default value empty string.

        Output: Memberships
        The output should strictly adhere to the following JSON schema:
        {
          "Memberships": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "Organization_name": { "type": "string", "default": "" },
                "Membership_title": { "type": "string", "default": "" },
                "Start_Year": { "type": "string", "default": "" },
                "Start_Month": { "type": "string", "default": "" },
                "End_Year": { "type": "string", "default": "" },
                "End_Month": { "type": "string", "default": "" }
              },
              "required": ["Organization_name", "Membership_title"]
            }
          }
        }
    """
    result = generate_response(Input, system_message)
    cleaned_response_text = result.replace('```json', '').replace('```', '').strip()
    print("Cleaned response text:")
    print(cleaned_response_text)
    data.append(cleaned_response_text)
    return result


def Certifications_courses(Input):
    system_message = """
        You are tasked with analyzing the provided Input and extracting relevant text values for specified entities.
        Follow the given JSON schema accurately to ensure all details are correctly captured don't skip any Information.
        If a field is not present, use the default value empty string.

        Output: Certifications_courses
        The output should strictly adhere to the following JSON schema:
        {
          "Certifications_courses": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "Certification_name": { "type": "string", "default": "" },
                "Issuing_organization": { "type": "string", "default": "" },
                "Year_of_issuance": { "type": "string", "default": "" }
              },
              "required": ["Certification_name", "Issuing_organization"]
            }
          }
        }
    """
    result = generate_response(Input, system_message)
    cleaned_response_text = result.replace('```json', '').replace('```', '').strip()
    print("Cleaned response text:")
    print(cleaned_response_text)
    data.append(cleaned_response_text)
    return result


def Employement_Summary(Input):
    system_message = """
        You are tasked with analyzing the provided Input and extracting relevant text values for specified entities.
        Follow the given JSON schema accurately to ensure all details are correctly captured don't skip any Information.
        If a field is not present, use the default value empty string.

        Output: CV_Summary
        The output should strictly adhere to the following JSON schema:
        {
          "CV_Summary": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "Start_Year": { "type": "string", "default": "" },
                "End_Year": { "type": "string", "default": "" },
                "Company_name": { "type": "string", "default": "" },
                "Description": { "type": "string", "default": "" },
                "Positions": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "Position_title": { "type": "string", "default": "" },
                      "Start_Year": { "type": "string", "default": "" },
                      "End_Year": { "type": "string", "default": "" }
                    },
                    "required": ["Position_title", "Start_Year", "End_Year"]
                  }
                }
              },
              "required": ["Start_Year", "End_Year", "Company_name", "Positions", "Description"]
            }
          }
        }
    """
    result = generate_response(Input, system_message)
    cleaned_response_text = result.replace('```json', '').replace('```', '').strip()
    print("Cleaned response text:")
    print(cleaned_response_text)
    data.append(cleaned_response_text)
    return result


def Personal_Information(Input):
    system_message = """
        You are tasked with analyzing the provided Input and extracting relevant text values for specified entities.
        Follow the given JSON schema accurately to ensure all details are correctly captured don't skip any Information.
        If a field is not present, use the default value empty string.

        Output: Personal_Information
        The output should strictly adhere to the following JSON schema:
        {
            "Candidate_name": { "type": "string", "default": "" },
            "Date_of_birth": { "type": "string", "default": "" },
            "Nationality": { "type": "string", "default": "" },
            "Marital_status": { "type": "string", "default": "" },
            "Residence": { "type": "string", "default": "" },
            "LinkedIn_profile": { "type": "string", "default": "" },
            "Personal_website": { "type": "string", "default": "" },
            "Contact_number": { "type": "string", "default": "" },
            "Email": { "type": "string", "default": "" },
            "Current_company": { "type": "string", "default": "" },
            "Current_position": { "type": "string", "default": "" },
            "Highest_Education": {
              "type": "object",
              "properties": {
                "Institution": { "type": "string", "default": "" },
                "Degree": { "type": "string", "default": "" },
                "End_Year": { "type": "string", "default": "" }
              },
              "required": ["Institution", "Degree", "End_Year"]
            },
            "Personal_Profile": { "type": "string", "default": "" },
            "Education_details": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "Institution": { "type": "string", "default": "" },
                  "Location": { "type": "string", "default": "" },
                  "Degree": { "type": "string", "default": "" },
                  "Start_Year": { "type": "string", "default": "" },
                  "End_Year": { "type": "string", "default": "" }
                }
              }
            }
        }
    """
    result = generate_response(Input, system_message)
    cleaned_response_text = result.replace('```json', '').replace('```', '').strip()
    print("Cleaned response text:")
    print(cleaned_response_text)
    data.append(cleaned_response_text)
    return result


response = continuous_chat(client, assistant_id, Input, thread)
print("FINAL RESPONSE:",response)
