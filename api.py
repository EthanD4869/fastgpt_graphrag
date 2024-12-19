from xinference.client import Client
import psycopg2
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
import re
import requests
import json
import time
import csv
import uuid
import os
import hashlib
import numpy as np
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

URL = None
API_KEY = None
MAX_DEPTH = 1

def generate_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def compute_similarity(query_vector, target_vector):
    """
    计算两个向量的余弦相似度。
    
    :param query_vector: 查询向量 (list or np.array)
    :param target_vector: 数据库中存储的目标向量 (list, np.array, or str)
    :return: 余弦相似度 (float)
    """
    # 将输入向量统一为 numpy 数组
    try:
        query_vector = np.array(query_vector, dtype=float)
        
        # 如果 target_vector 是字符串，尝试解析
        if isinstance(target_vector, str):
            import json
            target_vector = np.array(json.loads(target_vector), dtype=float)
        else:
            target_vector = np.array(target_vector, dtype=float)
    except Exception as e:
        print(f"Error converting vectors: {e}")
        return 0.0  # 无法转换时返回最低相似度

    # 检查向量的维度是否一致
    if query_vector.shape != target_vector.shape:
        print(f"Dimension mismatch: query_vector {query_vector.shape}, target_vector {target_vector.shape}")
        return 0.0

    # 计算向量的余弦相似度
    dot_product = np.dot(query_vector, target_vector)
    query_norm = np.linalg.norm(query_vector)
    target_norm = np.linalg.norm(target_vector)

    # 避免除以零
    if query_norm == 0 or target_norm == 0:
        return 0.0

    return dot_product / (query_norm * target_norm)

def count_entity_embeddings():
    try:
        # 建立数据库连接
        with psycopg2.connect(
            host=pg_host, port=pg_port, database=pg_database, user=pg_user, password=pg_password
        ) as conn:
            with conn.cursor() as cursor:
                # 执行计数查询
                cursor.execute("SELECT COUNT(*) FROM entity_embeddings;")
                count = cursor.fetchone()[0]
                print(f"entity_embeddings 表中共有 {count} 条数据。")
                return count
    except Exception as e:
        print(f"查询 entity_embeddings 表数据数量时发生错误：{e}")

def count_edges_for_entity(entity_name):
    with connection_pool.session_context('root', 'nebula') as session:
        # 切换到目标空间
        session.execute('USE relationship')

        # 查询并计数与 entityA 相关的所有边
        query = f'MATCH (a:entity)-[r:relation]->() WHERE a.name == "{entity_name}" RETURN COUNT(r) AS edge_count'
        
        # 执行查询
        resp = session.execute(query)
        if resp.is_succeeded():
            edge_count = resp.rows()[0]["edge_count"]
            print(f"Entity '{entity_name}' has {edge_count} edges.")
            return edge_count
        else:
            print(f"Failed to count edges for entity '{entity_name}': {resp.error_msg()}")
            return 0

def gpt_request(content, max_retries=20, retry_delay=5):
    for attempt in range(max_retries):
        try:
            url = URL
            api_key = API_KEY  # Replace with your OpenAI API key
            data = {
                "model": "qwen-plus",  # Correct model name
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.01
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            print("Sending request...")
            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                res = response.json()
                return res
            else:
                print(f"Request failed with status code {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed with exception: {e}")

        time.sleep(retry_delay)

    print(f"Failed after {max_retries} attempts.")
    return None

# bge-m3
client = Client("http://192.168.0.131:9997")
embedding_model = client.get_model("bge-m3")  # 嵌入模型

# PostgreSQL 连接参数
pg_host = "192.168.0.131"
pg_port = "5432"
pg_database = "knowledge_graph"
pg_user = "admin"
pg_password = "1234"

# 1. 初始化 Nebula 数据库连接
config = Config()
config.max_connection_pool_size = 10
connection_pool = ConnectionPool()
ok = connection_pool.init([('192.168.0.131', 9669)], config)

# 2. 创建 nebula graph space
def create_space(space_name):
    with connection_pool.session_context('root', 'nebula') as session:
        # 创建空间
        create_space_query = f'CREATE SPACE IF NOT EXISTS `{space_name}` (vid_type = FIXED_STRING(128));'
        resp = session.execute(create_space_query)
        
        # 检查是否创建成功
        if resp.is_succeeded():
            print(f"Space '{space_name}' created or already exists.")
        else:
            print(f"Failed to create space '{space_name}': {resp.error_msg()}")

# 3. 创建 PostgreSQL 表
def create_tables():
    with psycopg2.connect(host=pg_host, port=pg_port, database=pg_database, user=pg_user, password=pg_password) as conn:
        with conn.cursor() as cursor:
            # 初始化 vector 扩展
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 创建实体嵌入表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entity_embeddings (
                    entity VARCHAR PRIMARY KEY,
                    embedding FLOAT8[]
                );
            """)

            # 将 embedding 列的类型更改为 vector(1024)
            cursor.execute("ALTER TABLE entity_embeddings ALTER COLUMN embedding TYPE vector(1024);")
            
            # 创建实体文本表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entity_texts (
                    entity VARCHAR,
                    text TEXT,
                    FOREIGN KEY (entity) REFERENCES entity_embeddings(entity)
                );
            """)
            
            # 创建文本嵌入表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_embeddings (
                    text_hash CHAR(32) PRIMARY KEY,
                    embedding vector(1024) NOT NULL
                );
            """)

            # 提交更改
            conn.commit()

# 4. 文件分段
def split_file(file_path):
    """
    从 CSV 文件的 A 列读取数据，从第二行开始，将每行内容作为一条记录。

    :param file_path: str, CSV 文件的路径
    :return: list, 包含 A 列每行数据的列表
    """
    segments = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # 跳过表头
            for row in csv_reader:
                # 假设 A 列是第一列
                if len(row) > 0:  # 确保当前行有数据
                    segments.append(row[0])  # 读取 A 列数据
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
    return segments

# 5. 提取三元组
def extract_triplets(segment, max_knowledge_triplets = 10):
    prompt = (
        "以下是一些文本内容。根据文本内容，提取最多 "
        f"{max_knowledge_triplets} "
        "个知识三元组，格式为（主体，谓词，客体）。请避免使用停用词。\n"
        "如果无法从文本中确定关系或客体，可以用 'None' 占位符替代不确定的元素。\n"
        "---------------------\n"
        "示例:"
        "文本：Alice 是 Bob 的母亲。"
        "三元组：\n(Alice, 是母亲, Bob)\n"
        "文本：Philz 是一家咖啡店，成立于1982年，位于伯克利。\n"
        "三元组：\n"
        "(Philz, 是, 咖啡店)\n"
        "(Philz, 成立于, 伯克利)\n"
        "(Philz, 成立于, 1982年)\n"
        "文本：小明在杭州吗？\n"
        "三元组：\n"
        "(小明, 在, 杭州)\n"
        "文本：你知道北京在哪里吗？\n"
        "三元组：\n"
        "(北京, 在, None)\n"
        "---------------------\n"
        f"文本：{segment}\n"
        "三元组：\n"
    )
    response = gpt_request(prompt)
    triplets_text = response['choices'][0]['message']['content']
    
    # 使用正则表达式提取三元组
    triplet_pattern = r"\(([^,]+), ([^,]+), ([^)]+)\)"
    matches = re.findall(triplet_pattern, triplets_text)
    
    triplets = []
    for match in matches:
        entity1, relation, entity2 = match
        triplets.append((entity1.strip(), relation.strip(), entity2.strip()))
    
    return triplets

# 6. 存储实体嵌入到 PostgreSQL
def store_entity_embedding(entity, embedding_vector):
    with psycopg2.connect(host=pg_host, port=pg_port, database=pg_database, user=pg_user, password=pg_password) as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute("INSERT INTO entity_embeddings (entity, embedding) VALUES (%s, %s)", (entity, embedding_vector))
            except psycopg2.errors.UniqueViolation:
                print(f"实体 '{entity}' 已存在，跳过插入。")

# 7. 存储实体-原始文本的表
def store_entity_text(entity, original_text):
    with psycopg2.connect(host=pg_host, port=pg_port, database=pg_database, user=pg_user, password=pg_password) as conn:
        with conn.cursor() as cursor:
            # 检查实体和文本是否已存在
            cursor.execute("SELECT 1 FROM entity_texts WHERE entity = %s AND text = %s", (entity, original_text))
            exists = cursor.fetchone()
            
            # 仅在实体和文本都不同时插入
            if not exists:
                cursor.execute("INSERT INTO entity_texts (entity, text) VALUES (%s, %s)", (entity, original_text))
            else:
                print(f"实体 '{entity}' 和文本已存在，跳过插入。")

# 8. 存储文本及其嵌入向量
def store_text_embedding(text, embedding_vector):
    text_hash = generate_hash(text)  # 生成哈希值
    with psycopg2.connect(host=pg_host, port=pg_port, database=pg_database, user=pg_user, password=pg_password) as conn:
        with conn.cursor() as cursor:
            try:
                # 检查文本是否已存在
                cursor.execute("SELECT 1 FROM text_embeddings WHERE text_hash = %s", (text_hash,))
                exists = cursor.fetchone()
                
                if not exists:
                    # 插入新记录
                    cursor.execute(
                        "INSERT INTO text_embeddings (text_hash, embedding) VALUES (%s, %s)", 
                        (text_hash, embedding_vector)
                    )
                else:
                    print(f"文本已存在，跳过插入")
            except Exception as e:
                print(f"存储文本嵌入时发生错误：{e}")

# 9. 存储三元组到 Nebula
def store_triplet(triplet):

    # 确保传入的三元组是 (entity1, relation, entity2) 格式
    if len(triplet) != 3:
        print(f"错误: 三元组格式不正确: {triplet}")
        return

    # 解包三元组
    entity1, relation, entity2 = triplet
    
    # 存储到 Nebula 数据库
    with connection_pool.session_context('root', 'nebula') as session:
        # 空间名称
        space_name = 'relationship'

        # 确保切换到正确的空间
        session.execute(f'USE {space_name}')
        
        # 创建标签
        create_tag_query = 'CREATE TAG IF NOT EXISTS entity(name string);'
        tag_resp = session.execute(create_tag_query)
        assert tag_resp.is_succeeded(), tag_resp.error_msg()
        if not tag_resp.is_succeeded():
            print(f"Failed to create tag 'entity': {tag_resp.error_msg()}")
        else:
            print("Tag 'entity' created successfully or already exists.")

        # 创建边类型
        create_edge_query = 'CREATE EDGE IF NOT EXISTS relation();'
        edge_resp = session.execute(create_edge_query)
        if not edge_resp.is_succeeded():
            print(f"Failed to create edge 'relation': {edge_resp.error_msg()}")
        else:
            print("Edge 'relation' created successfully or already exists.")

        # 插入实体顶点1
        insert_vertex1_query = f'INSERT VERTEX entity(name) VALUES "{entity1}":("{entity1}")'
        vertex1_resp = session.execute(insert_vertex1_query)
        if not vertex1_resp.is_succeeded():
            print(f"Failed to insert vertex '{entity1}': {vertex1_resp.error_msg()}")
        else:
            print(f"Vertex '{entity1}' inserted successfully or already exists.")

        # 插入实体顶点2
        insert_vertex2_query = f'INSERT VERTEX entity(name) VALUES "{entity2}":("{entity2}")'
        vertex2_resp = session.execute(insert_vertex2_query)
        if not vertex2_resp.is_succeeded():
            print(f"Failed to insert vertex '{entity2}': {vertex2_resp.error_msg()}")
        else:
            print(f"Vertex '{entity2}' inserted successfully or already exists.")

        # 插入关系边
        insert_edge_query = f'INSERT EDGE relation() VALUES "{entity1}"->"{entity2}":()'
        edge_resp = session.execute(insert_edge_query)
        if not edge_resp.is_succeeded():
            print(f"Failed to insert edge '{entity1}->{relation}->{entity2}': {edge_resp.error_msg()}")
        else:
            print(f"Edge '{entity1}->{relation}->{entity2}' inserted successfully.")

# 10. 在 Nebula 图数据库中找到相似实体
def find_similar_entities_in_nebula(similar_results, depth=1):
    found_entities = set(similar_results)  # 使用集合来避免重复
    with connection_pool.session_context('root', 'nebula') as session:
        # 切换到使用的空间
        session.execute('USE relationship')

        # 遍历当前的相似实体结果列表 
        for entity in similar_results:
            escaped_entity = entity.replace('"', '\\"')
            query = f"""
            GO {depth} STEPS FROM "{escaped_entity}" OVER relation YIELD dst(edge) AS connected_entity
            """
            resp = json.loads(session.execute_json(query).decode('utf-8'))

            connected_entities = [entry['row'][0] for entry in resp['results'][0]['data']]
            
            # 处理查询结果，提取新找到的实体并加入到found_entities中
            for connected_entity in connected_entities:
                found_entities.add(connected_entity)

    return list(found_entities)  # 返回包含所有找到的实体的列表

# 11. 查找与输入实体相关的其他实体，直到指定的最大深度。
def find_related_entities(similar_entities, max_depth):
    all_related_entities = set(similar_entities)  # 使用集合来避免重复
    
    # 对每个深度进行查询
    for depth in range(1, max_depth + 1):
        # print(f"正在查找深度 {depth} 的相关实体...")
        related_entities_at_depth = find_similar_entities_in_nebula(similar_entities, depth)
        # print(f"depth {depth} 的相关实体:", related_entities_at_depth)
        
        # 合并结果
        all_related_entities.update(related_entities_at_depth)  # 使用update来合并
    
    return list(all_related_entities)

# 12. 提问提取三元组
def extract_triplet_from_question(question):
    return extract_triplets(question)

# 13. 找到相似实体
def find_similar_entity(embedding_vector):
    with psycopg2.connect(host=pg_host, port=pg_port, database=pg_database, user=pg_user, password=pg_password) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT entity
                FROM entity_embeddings
                ORDER BY embedding <-> %s::vector
                LIMIT 1
            """, (embedding_vector,))  # 直接传递 embedding_vector 列表
            return cursor.fetchone()  # 返回最相似的一个实体

# 14. 找到原始文本
def find_original_texts(entities):
    texts = []
    with psycopg2.connect(host=pg_host, port=pg_port, database=pg_database, user=pg_user, password=pg_password) as conn:
        with conn.cursor() as cursor:
            for entity in entities:
                cursor.execute("SELECT text FROM entity_texts WHERE entity = %s", (entity,))
                texts.extend(cursor.fetchall())
    return texts

# 15. 使用大模型回答
def generate_answer(original_texts):
    combined_text = ' '.join([text[0] for text in original_texts])  # 处理文本格式
    response = gpt_request(combined_text)
    return response['choices'][0]['message']['content']

# 16
def estimate_token_count(text):
    # 中文每个字符算一个 token，英文按空格分词
    chinese_tokens = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    english_tokens = sum(len(word) for word in text.split() if not '\u4e00' <= word <= '\u9fff')
    return chinese_tokens + english_tokens

# 17
def process_question(question):
    """
    处理问题并返回与问题相关的原始文本。

    参数:
        question (str): 用户输入的问题。

    返回:
        list: 与问题相关的原始文本列表。
    """
    # 提取问题中的三元组
    triplet_from_question = extract_triplet_from_question(question)
    
    # 对问题三元组中的每个实体进行嵌入计算
    question_embeddings = []
    for triplet in triplet_from_question:
        for i, entity in enumerate(triplet):
            entity = entity.strip()
            
            # 仅计算主体或客体（跳过关系部分）且忽略 None 或空字符串
            if i != 1 and entity and entity.lower() != 'none':
                # 对每个实体计算嵌入
                m3e_response = embedding_model.create_embedding(entity)
                embedding_vector = m3e_response['data'][0]['embedding']
                question_embeddings.append((entity, embedding_vector))

    # 对每个实体嵌入执行相似实体查询
    similar_entities = []
    for entity, embedding_vector in question_embeddings:
        similar_results = find_similar_entity(embedding_vector)
        similar_entities.extend(similar_results)

    # 查找并组合相关的原始文本
    related_entities = find_related_entities(similar_entities, MAX_DEPTH)

    original_texts = find_original_texts(related_entities)
    original_texts = [item[0] for item in original_texts]
    original_texts = list(set(original_texts))
    
    return original_texts

def fetch_embedding_from_pg(text_hash):
    """
    根据 text_hash 从 PostgreSQL 中获取对应的嵌入向量。

    参数:
        text_hash (str): 文本的哈希值。

    返回:
        list: 对应的嵌入向量，如果未找到返回 None。
    """
    try:
        with psycopg2.connect(host=pg_host, port=pg_port, database=pg_database, user=pg_user, password=pg_password) as conn:
            with conn.cursor() as cursor:
                # 查询对应的嵌入向量
                cursor.execute("SELECT embedding FROM text_embeddings WHERE text_hash = %s", (text_hash,))
                result = cursor.fetchone()
                if result:
                    return result[0]  # 返回嵌入向量
                else:
                    print(f"未找到 text_hash 为 {text_hash} 的记录。")
                    return None
    except Exception as e:
        print(f"查询嵌入向量时发生错误：{e}")
        return None

def model_call(question):
    try:
        # 获取与问题相关的文本列表
        quote_list = process_question(question)

        # 处理文本片段超过 10 条的情况
        if len(quote_list) > 10:
            # 获取问题的嵌入向量
            query = question.strip()
            query_m3e_response = embedding_model.create_embedding(query)
            query_embedding_vector = query_m3e_response['data'][0]['embedding']

            # 获取文本片段的向量
            quote_vectors = []
            for quote in quote_list:
                quote = quote.strip()
                text_hash = generate_hash(quote)
                embedding_vector = fetch_embedding_from_pg(text_hash)

                if embedding_vector is not None:
                    quote_vectors.append((quote, embedding_vector))
                else:
                    print(f"警告：未找到文本的向量，跳过该文本: {quote}")

            if not quote_vectors:
                return {"error": "未找到任何有效的文本嵌入，无法生成答案。"}

            # 计算相似度并排序
            similarities = [
                (quote, compute_similarity(query_embedding_vector, vector))
                for quote, vector in quote_vectors
            ]
            sorted_quotes = sorted(similarities, key=lambda x: x[1], reverse=True)
            quote_list = [quote[0] for quote in sorted_quotes[:10]]

        # 构建生成回答的提示词
        prompt = f"""
        问题: {question}

        以下是与问题相关的原始文本片段：

        {quote_list}

        请根据以上信息，回答问题:
        """
        # 调用生成回答的模型
        answer = generate_answer(prompt)
        return {"answer": answer, "quote_list": quote_list}
    except Exception as e:
        return {"error": str(e)}

@app.route('/model_call', methods=['POST'])
def model_call_api():
    try:
        # 获取请求体数据
        data = request.get_json()
        if not data:
            return Response(json.dumps({"error": "No input data provided"}), 
                            status=400, 
                            mimetype='application/json; charset=utf-8')

        question = data.get("question")
        if not question:
            return Response(json.dumps({"error": "'question' is required."}), 
                            status=400, 
                            mimetype='application/json; charset=utf-8')

        # 调用 `model_call` 函数
        result = model_call(question)

        # 如果函数返回错误信息，返回 500 状态码
        if "error" in result:
            return Response(json.dumps(result), 
                            status=500, 
                            mimetype='application/json; charset=utf-8')

        # 返回生成的答案
        return Response(json.dumps(result, ensure_ascii=False), 
                        status=200, 
                        mimetype='application/json; charset=utf-8')

    except Exception as e:
        return Response(json.dumps({"error": str(e)}, ensure_ascii=False), 
                        status=500, 
                        mimetype='application/json; charset=utf-8')

# 19. 整体流程
def main(csv_file_path, question):
    # 记录整体流程开始时间
    start_time = time.time()
    
    # 创建 Nebula 空间
    space_start = time.time()
    create_space('relationship')
    print(f"创建 Nebula 空间耗时: {time.time() - space_start:.2f} 秒")
    
    # 创建 PostgreSQL 表
    table_start = time.time()
    create_tables()
    print(f"创建 PostgreSQL 表耗时: {time.time() - table_start:.2f} 秒")
    
    # 分段处理
    segment_start = time.time()
    segments = split_file(csv_file_path)
    print(f"文本分段耗时: {time.time() - segment_start:.2f} 秒")
    print(f"文本分段数量: {len(segments)}")
    
    # 处理每段文本
    processing_start = time.time()
    for i, segment in enumerate(segments):
        print(f"正在处理第 {i+1}/{len(segments)} 段文本...")
        segment_start = time.time()
        
        triplets = extract_triplets(segment)  # 提取三元组列表
        for entity1, relation, entity2 in triplets:
            # 存储三元组到 Nebula
            store_triplet((entity1, relation, entity2))
            
            for entity in [entity1, entity2]:
                if entity != "None":  # 忽略占位符
                    entity = entity.strip()
                    embedding_start = time.time()
                    
                    # 嵌入模型生成向量
                    entity_m3e_response = embedding_model.create_embedding(entity)
                    entity_embedding_vector = entity_m3e_response['data'][0]['embedding']
                    
                    # 存储嵌入和原始文本
                    store_entity_embedding(entity, entity_embedding_vector)
                    store_entity_text(entity, segment)
                
                print(f"实体 '{entity}' 嵌入生成与存储耗时: {time.time() - embedding_start:.2f} 秒")

        segment = segment.strip()
        segment_m3e_response = embedding_model.create_embedding(segment)
        segment_embedding_vector = segment_m3e_response['data'][0]['embedding']
        store_text_embedding(segment, segment_embedding_vector)
        
        print(f"处理第 {i+1} 段文本耗时: {time.time() - segment_start:.2f} 秒")
    
    print(f"所有段落处理耗时: {time.time() - processing_start:.2f} 秒")
    
    count_entity_embeddings()

# 调用
if __name__ == "__main__":
    # file_path = 'fastgpt_dataset.csv'  # 你的文件路径
    # main(file_path, question)
    app.run(host='0.0.0.0', port=6026)

# 关闭连接池
connection_pool.close()
