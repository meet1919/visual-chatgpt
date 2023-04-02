from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# initialize HF LLM
flan_t5 = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature":1e-10}
)

# build prompt template for simple question-answering
template = """Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(
    prompt=prompt,
    llm=flan_t5
)

question = "what is square root of 4"

print(llm_chain.run(question))