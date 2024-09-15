import os
import json  # Import json for parsing
from dotenv import load_dotenv  # Import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

class CodeOut(BaseModel):
    code: str = Field(description="The generated code")
    test: str = Field(description="Test cases for the generated code")
    comment: str = Field(description="Explanation of the generated code")

class ResponseGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        # No need to store the API key since it's automatically picked up

    def get_response(self, sys_prompt: str, usr_prompt: str) -> str:
        llm = ChatOpenAI(model=self.model, temperature=0)  # No need to pass the API key
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{user_prompt}")
        ])
        chain = prompt | llm
        ai_msg = chain.invoke({
            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt
        })
        return ai_msg.content

    def get_structured_response(self, sys_prompt: str, usr_prompt: str, schema: BaseModel) -> BaseModel:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{user_prompt}")
        ])
        llm = ChatOpenAI(temperature=0, model=self.model)
        structured_chain = prompt | llm.with_structured_output(schema)  # Use structured output
        ai_msg = structured_chain.invoke({
            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt
        })
        return schema.parse_obj(ai_msg.dict())  # Return the structured response as an instance of the schema

if __name__ == "__main__":
    # Test the ResponseGenerator
    rg = ResponseGenerator()
    sys_prompt = "You are a helpful assistant."
    usr_prompt = "Can you provide a summary of the benefits of using AI in healthcare?"
    
    structured_response = rg.get_structured_response(sys_prompt, usr_prompt, CodeOut)
    print("Structured Response from AI:")
    print(structured_response)

