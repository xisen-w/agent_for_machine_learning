from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

class AdvancedCodeWriter:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    class CodeOut(BaseModel):
        code: str = Field(description="The generated code")
        test: str = Field(description="Test cases for the generated code")
        comment: str = Field(description="Explanation of the generated code")

    def write_code(self, demand: str, knowledge_base: str) -> dict:
        sys_prompt = '''
        You are an advanced software engineer that writes useful and correct code...
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

    def advanced_code_writer_v2(self, demand, knowledge_base, max_iter=6):
        context = ""
        code_out = self.write_code(demand, knowledge_base)
        code, test, comment = code_out['code'], code_out['test'], code_out['comment']
        i = 0
        state = self.test_code(code, test)

        while state != "True" and i < max_iter:
            context += f"Iteration {i}: Error Notice: {state}. Previous code explanation: {comment}\n"
            debug_output = CodeDebugger(self.model).debug_code(f"{code}\n{test}", context)
            code, test, comment = debug_output['corrected_code'], debug_output['test'], debug_output['explanation']
            i += 1
            state = self.test_code(code, test)

        return code, test, comment

    @staticmethod
    def test_code(code: str, test: str) -> str:
        combined_code = f"{code}\n{test}"
        result = CodeExecutor.execute_code(combined_code)
        if result.startswith("Traceback"):
            return result
        return "True"