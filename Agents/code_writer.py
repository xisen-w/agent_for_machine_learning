from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import io
import contextlib
import traceback
from utils.response_generator import ResponseGenerator  # Add this import

class CodeWriter:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    class CodeOut(BaseModel):
        code: str = Field(description="The generated code")
        test: str = Field(description="Test cases for the generated code")
        comment: str = Field(description="Explanation of the generated code")

    def write_code(self, demand: str, knowledge_base: str) -> dict:
        sys_prompt = '''
        You are an advanced software engineer that writes useful and correct code.
        Given a specific demand and a knowledge base, generate code and a script to run it.
        Provide the code, test cases, and explanation for the code in JSON format.
        '''
        usr_prompt = f"Demand: {demand}\nKnowledge Base: {knowledge_base}\n"
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{user_prompt}")
        ])
        llm = ChatOpenAI(temperature=0, model=self.model)
        code_gen_chain = prompt | llm.with_structured_output(self.CodeOut)
        ai_msg = code_gen_chain.invoke({
            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt
        })
        return ai_msg.dict()

    def debug(self, code: str, error: str) -> dict:
        usr_prompt = f"Code:\n{code}\n\nError Description:\n{error}\n"
        sys_prompt = '''
        You are a coding assistant with expertise in software engineering.
        Debug the code and provide the corrected code, test cases, and an explanation.
        '''
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{user_prompt}")
        ])
        llm = ChatOpenAI(temperature=0, model=self.model)
        debug_chain = prompt | llm.with_structured_output(self.CodeOut)
        ai_msg = debug_chain.invoke({
            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt
        })
        return ai_msg.dict()

    @staticmethod
    def test_coder(code: str, test: str) -> str:
        combined_code = f"{code}\n{test}"
        result = CodeWriter.execute_code(combined_code)
        if result.startswith("Traceback"):
            return result
        return "True"

    @staticmethod
    def execute_code(code: str) -> str:
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(code)
        except Exception as e:
            return traceback.format_exc()
        return output.getvalue().strip()

    def advanced_writing(self, demand: str, knowledge_base: str) -> dict:
        code_out = self.write_code(demand, knowledge_base)
        code, test, comment = code_out['code'], code_out['test'], code_out['comment']
        state = self.test_coder(code, test)
        if state == "True":
            return code, test, comment
        return self.debug(code, state)

    def advanced_writing_v2(self, demand: str, knowledge_base: str, max_iter: int = 6) -> dict:
        context = ""
        code_out = self.write_code(demand, knowledge_base)
        code, test, comment = code_out['code'], code_out['test'], code_out['comment']
        i = 0
        state = self.test_coder(code, test)

        while state != "True" and i < max_iter:
            context += f"Iteration {i}: Error Notice: {state}. Previous code explanation: {comment}\n"
            debug_output = self.debug(f"{code}\n{test}", context)
            code, test, comment = debug_output['corrected_code'], debug_output['test'], debug_output['explanation']
            i += 1
            state = self.test_coder(code, test)

        return {"code": code, "test": test, "comment": comment}