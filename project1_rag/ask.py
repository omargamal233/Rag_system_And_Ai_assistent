from rag_chain import rag_chain
def ask_question(query:str):
    result=rag_chain.invoke(query)
    return result


result=ask_question("I have a deadline, which framework should I use to build premium data visualizations?")
print(result)