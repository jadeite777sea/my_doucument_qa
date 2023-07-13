
import openai
import faiss
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import os
import json


def get_current_identity(name):
    """获取人物信息"""
    with open('./shenfen.txt', 'r', encoding='utf-8') as f:
            shenfen_data = f.readline() 
    shenfen_data=shenfen_data.split(' ')
    identity_info = {
        "name": name,
        "战场等级": shenfen_data[1],
        "战绩等级": shenfen_data[2],
        "军衔等级": shenfen_data[3],
    }
    return json.dumps(identity_info)


def create_embeddings(input):
    """Create embeddings for the provided input."""
    result = []
    # limit about 1000 tokens per request
    lens = [len(text) for text in input]
    query_len = 0
    start_index = 0
    tokens = 0

    def get_embedding(input_slice):
        embedding = openai.Embedding.create(model="text-embedding-ada-002", input=input_slice)
        return [(text, data.embedding) for text, data in zip(input_slice, embedding.data)], embedding.usage.total_tokens

    for index, l in tqdm(enumerate(lens)):
        query_len += l
        if query_len > 4096:
            ebd, tk = get_embedding(input[start_index:index + 1])
            query_len = 0
            start_index = index + 1
            tokens += tk
            result.extend(ebd)

    if query_len > 0:
        ebd, tk = get_embedding(input[start_index:])
        tokens += tk
        result.extend(ebd)
    return result, tokens

def create_embedding(text):
    """Create an embedding for the provided text."""
    embedding = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return text, embedding.data[0].embedding

class QA():
    def __init__(self,data_embe) -> None:
        d = 1536
        index = faiss.IndexFlatL2(d)
        embe = np.array([emm[1] for emm in data_embe])
        data = [emm[0] for emm in data_embe]
        index.add(embe)
        self.index = index
        self.data = data
    def __call__(self, query):
        embedding = create_embedding(query)
        context = self.get_texts(embedding[1], limit)
        answer = self.completion(query,context)
        return answer,context

    def get_texts(self,embeding,limit):
        _,text_index = self.index.search(np.array([embeding]),limit)
        context = []
        for i in list(text_index[0]):
            context.extend(self.data[i:i+5])
        return context
    
    def completion(self,query, context):
        """Create a completion."""
        lens = [len(text) for text in context]

        maximum = 3000
        for index, l in enumerate(lens):
            maximum -= l
            if maximum < 0:
                context = context[:index + 1]
                print("超过最大长度，截断到前", index + 1, "个片段")
                break
        

        text = "\n".join(f"{index}. {text}" for index, text in enumerate(context))
        messages=[
                {'role': 'system',
                'content': f'你是一个有帮助的AI游戏问答助手，从下文中提取有用的内容进行回答，一定要考虑我的个人信息，不能回答不在下文提到的内容，相关性从高到底排序,：\n\n{text}'},
                {"role": "assistant", "content":"好的,我记住了你的游戏名字是zhangyan"},
                {'role': 'user', 'content': query},
            ]
        #设置对应的可用函数
        functions = [
            {
                "name": "get_current_identity",
                "description": "获取个人信息：战场等级，战绩等级，军衔等级",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "名字,例如：zhangyan",
                        },
                    },
                    "required": ["name"],
                },
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call="auto", #function_call默认值即为auto，这里进行显示说明
        )
        response_message =response["choices"][0]["message"]
        # 步骤2：检查GPT是否想要调用一个函数
        if response_message.get("function_call"):
            # 步骤3：调用函数
            # 注意：JSON响应可能并非总是有效的；要确保处理错误
            available_functions = {
                "get_current_identity": get_current_identity,
            }  # 这个示例中只有一个函数，但你可以有多个
            function_name = response_message["function_call"]["name"]
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = fuction_to_call(
                name=function_args.get("name"),
            )
            # 步骤4：将函数调用和函数响应的信息发送给GPT
            # 将助手的回复加入对话
            messages.append(response_message)  
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  # 将函数响应加入对话
            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
            )  # 从GPT获取一个新的响应，这个响应可以看到函数的响应
            return second_response.choices[0].message.content
        #同时也存在没有调用函数的情况
        else:
            return response.choices[0].message.content


if __name__ == '__main__':
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")


    # 修改接口地址
    openai.api_base = "https://api.f2gpt.com/v1"


    parser = argparse.ArgumentParser(description="Document QA")
    parser.add_argument("--input_file", default="input.txt", dest="input_file", type=str,help="输入文件路径")
    parser.add_argument("--file_embeding", default="input_embed.pkl", dest="file_embeding", type=str,help="文件embeding文件路径")
    parser.add_argument("--print_context", action='store_true',help="是否打印上下文")
    

    args = parser.parse_args()

    if os.path.isfile(args.file_embeding):
        data_embe = pickle.load(open(args.file_embeding,'rb'))
    else:
        with open(args.input_file,'r',encoding='utf-8') as f:
            texts = f.readlines()
            texts = [text.strip() for text in texts if text.strip()]
            data_embe,tokens = create_embeddings(texts)
            pickle.dump(data_embe,open(args.file_embeding,'wb'))
            print("文本消耗 {} tokens".format(tokens))

    qa =QA(data_embe)

    limit = 10
    while True:
        query = input("请输入查询(help可查看指令)：")
        if query == "quit":
            break
        elif query.startswith("limit"):
            try:
                limit = int(query.split(" ")[1])
                print("已设置limit为", limit)
            except Exception as e:
                print("设置limit失败", e)
            continue
        elif query == "help":
            print("输入limit [数字]设置limit")
            print("输入quit退出")
            continue
        answer,context = qa(query)
        if args.print_context:
            print("已找到相关片段：")
            for text in context:
                print('\t', text)
            print("=====================================")
        print("回答如下\n\n")
        print(answer.strip())
        print("=====================================")

