from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# 1. Khởi tạo model embedding
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

# 2. Kết nối tới Qdrant đang chạy trên máy local
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "migration_knowledge"

# 3. Chuẩn bị dữ liệu mẫu
documents = [
    {
        "id": 1,
        "text": "Some product variants failed to migrate because the destination platform has variant limits.",
        "platform": "shopify",
        "category": "product",
        "language": "en",
    },
    {
        "id": 2,
        "text": "Product migration can fail when option values are invalid or duplicated.",
        "platform": "shopify",
        "category": "product",
        "language": "en",
    },
    {
        "id": 3,
        "text": "Order migration may fail when the source data contains invalid customer references.",
        "platform": "woocommerce",
        "category": "order",
        "language": "en",
    },
    {
        "id": 4,
        "text": "Một số biến thể sản phẩm không được chuyển sang Shopify vì vượt quá giới hạn biến thể của nền tảng đích.",
        "platform": "shopify",
        "category": "product",
        "language": "vi",
    },
    {
        "id": 5,
        "text": "Giá trị thuộc tính sản phẩm không hợp lệ hoặc bị trùng lặp có thể khiến quá trình migration thất bại.",
        "platform": "shopify",
        "category": "product",
        "language": "vi",
    },
    {
        "id": 6,
        "text": "Lỗi kết nối API có thể khiến quá trình chuyển dữ liệu đơn hàng bị gián đoạn.",
        "platform": "woocommerce",
        "category": "order",
        "language": "vi",
    },
    {
        "id": 7,
        "text": "Images may fail to migrate if the source image URL is inaccessible or blocked by a firewall.",
        "platform": "magento",
        "category": "image",
        "language": "en",
    },
    {
        "id": 8,
        "text": "Customer records may be skipped when required fields are missing in the source database.",
        "platform": "woocommerce",
        "category": "customer",
        "language": "en",
    },
]

# 4. Tạo embedding cho toàn bộ tài liệu
texts = [doc["text"] for doc in documents]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
).tolist()

print("Number of documents:", len(embeddings))
print("Vector dimensions:", len(embeddings[0]))

# 5. Tạo collection
# Chỉ xóa collection này nếu đã tồn tại để chạy lại bài lab từ đầu.
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=len(embeddings[0]),
        distance=models.Distance.COSINE,
    ),
)

# 6. Insert dữ liệu vào Qdrant
points = []

for doc, vector in zip(documents, embeddings):
    points.append(
        models.PointStruct(
            id=doc["id"],
            vector=vector,
            payload={
                "text": doc["text"],
                "platform": doc["platform"],
                "category": doc["category"],
                "language": doc["language"],
            },
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True,
)

print("Inserted:", len(points), "points")

# 7. Kiểm tra số lượng point trong collection
info = client.get_collection(COLLECTION_NAME)
print("Collection status:", info.status)
print("Stored points:", info.points_count)