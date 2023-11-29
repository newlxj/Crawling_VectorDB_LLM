from Crawling import crawlData
import time
import tcvectordb
from tcvectordb.model.collection import Embedding
from tcvectordb.model.document import Document, Filter, SearchParams
from tcvectordb.model.enum import FieldType, IndexType, MetricType, EmbeddingModel, ReadConsistency
from tcvectordb.model.index import Index, VectorIndex, FilterIndex, HNSWParams, IVFFLATParams

tcvectordb.debug.DebugEnable = False
dbName='crawlingdb'
collectionName='tencent_knowledge'
class TencentVDB:
    def __init__(self, url: str, username: str, key: str, timeout: int = 30):
            """
            初始化客户端
            """
            # 创建客户端时可以指定 read_consistency，后续调用 sdk 接口的 read_consistency 将延用该值
            self._client = tcvectordb.VectorDBClient(url=url, username=username, key=key,
                                                    read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY, timeout=timeout)
 
    def create_db_and_collection(self):
        db = self._client.create_database(dbName)
        database_list = self._client.list_databases()
        for db_item in database_list:
            print(db_item.database_name)

        # 新建 Collection
        # 第一步，设计索引（不是设计表格的结构）
        # 1. 【重要的事】向量对应的文本字段不要建立索引，会浪费较大的内存，并且没有任何作用。
        # 2. 【必须的索引】：主键 id、向量字段 vector 这两个字段目前是固定且必须的，参考下面的例子；
        # 3. 【其他索引】：检索时需作为条件查询的字段，比如要按书籍的作者进行过滤，这个时候author字段就需要建立索引，
        #     否则无法在查询的时候对 author 字段进行过滤，不需要过滤的字段无需加索引，会浪费内存；
        # 4.  向量数据库支持动态 Schema，写入数据时可以写入任何字段，无需提前定义，类似 MongoDB.
        # 5.  例子中创建一个书籍片段的索引，例如书籍片段的信息包括 {id, vector, segment, bookName, page},
        #     id 为主键需要全局唯一，segment 为文本片段, vector 为 segment 的向量，vector 字段需要建立向量索引，假如我们在查询的时候要查询指定书籍
        #     名称的内容，这个时候需要对bookName建立索引，其他字段没有条件查询的需要，无需建立索引。
        # 6.  创建带 Embedding 的 collection 需要保证设置的 vector 索引的维度和 Embedding 所用模型生成向量维度一致，模型及维度关系：
        #     -----------------------------------------------------
        #             bge-base-zh                 ｜ 768
        #             m3e-base                    ｜ 768
        #             text2vec-large-chinese      ｜ 1024
        #             e5-large-v2                 ｜ 1024
        #             multilingual-e5-base        ｜ 768
        #     -----------------------------------------------------
        index = Index()
        index.add(VectorIndex('vector', 1024, IndexType.HNSW, MetricType.COSINE, HNSWParams(m=16, efconstruction=200)))
        index.add(FilterIndex('id', FieldType.String, IndexType.PRIMARY_KEY))
        index.add(FilterIndex('title', FieldType.String, IndexType.FILTER))
        ebd = Embedding(vector_field='vector', field='text', model=EmbeddingModel.TEXT2VEC_LARGE_CHINESE)

        # 第二步：创建 Collection
        # 创建支持 Embedding 的 Collection
        db.create_collection(
            name=collectionName,
            shard=3,
            replicas=0,
            description='爬虫向量数据库实验',
            index=index,
            embedding=ebd,
            timeout=50
        )

    def upsert_data(self):
        # 获取 Collection 对象
        db = self._client.database(dbName)
        coll = db.collection(collectionName)

        # upsert 写入数据，可能会有一定延迟
        # 1. 支持动态 Schema，除了 id、vector 字段必须写入，可以写入其他任意字段；
        # 2. upsert 会执行覆盖写，若文档 id 已存在，则新数据会直接覆盖原有数据(删除原有数据，再插入新数据)
        data = crawlData()
        docList =[]
        print(data)
        for i,dd in enumerate(data):
            docList =[]
            docList.append(Document(id=dd["url"],
            text=dd["text"],
            title=dd["title"]))
            coll.upsert(documents=docList,build_index=True)
            
        #     docList.append(Document(id=dd["url"],
        #                 text=dd["text"],
        #                 title=dd["title"]))
        # coll.upsert(documents=docList,build_index=True)
        # time.sleep(1)

    def clear(self):
        db = self._client.database(dbName)
        db.drop_database(dbName)

    def delete_and_drop(self):
        db = self._client.database(dbName)

        # 删除collection，删除collection的同时，其中的数据也将被全部删除
        db.drop_collection(collectionName)

        # 删除db，db下的所有collection都将被删除
        db.drop_database(dbName)



if __name__ == '__main__':
    test_vdb = TencentVDB('http://lb-p7oj0itn-j5gawvui3dz4lcn2.clb.ap-beijing.tencentclb.com:50000', key='lgrdYzlMsmc1GhVtcmilXhZevJBfFlId919EOvaE', username='root')
    test_vdb.clear()
    test_vdb.delete_and_drop()
    test_vdb.create_db_and_collection()
    test_vdb.upsert_data()

 