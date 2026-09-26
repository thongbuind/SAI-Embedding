# SAI-Embedding_100M — Mô hình embedding tiếng Việt chuyển đổi từ LLM **decoder-only** theo hướng LLM2Vec

**SAI-Embedding_100M** là mô hình biểu diễn câu và đoạn văn tiếng Việt, được xây dựng bằng cách chuyển mô hình ngôn ngữ [**SAI_100M**](https://huggingface.co/thongbuind/SAI_100M) thành một bộ mã hoá văn bản (text encoder) theo phương pháp **LLM2Vec**. Toàn bộ trọng số backbone được tái sử dụng từ SAI_100M, mô hình không được huấn luyện lại từ đầu và không thêm tầng chiếu (projection head) nào.

| Thuộc tính | Giá trị |
|---|---|
| Mô hình gốc | [`thongbuind/SAI_100M`](https://huggingface.co/thongbuind/SAI_100M) |
| Số tham số | ~114M |
| Ngôn ngữ | Tiếng Việt |
| Số chiều embedding | 768, cắt được còn 512 / 256 / 128 / 64 (Matryoshka) |
| Độ dài tối đa | 2.048 token (huấn luyện với passage 256–512 token) |
| Độ đo tương đồng | Cosine (vector đầu ra đã L2-normalize) |

## Tóm tắt

Các mô hình ngôn ngữ decoder-only được huấn luyện với mục tiêu dự đoán token tiếp theo và dùng causal attention, nên biểu diễn của mỗi token chỉ chứa thông tin từ các token phía trước. Tính chất này khiến chúng không phù hợp để dùng trực tiếp làm mô hình embedding. [LLM2Vec (BehnamGhader et al., 2024)](https://arxiv.org/abs/2404.05961) cho thấy một LLM decoder-only có thể được chuyển thành text encoder mạnh chỉ với vài bước thích nghi đơn giản.

Dự án này áp dụng hướng tiếp cận đó cho [SAI_100M](https://huggingface.co/thongbuind/SAI_100M), một LLM tiếng Việt cỡ nhỏ được huấn luyện từ đầu. Quy trình gồm hai giai đoạn:

1. Bật attention hai chiều và thích nghi mô hình bằng **masked next-token prediction (MNTP)**.
2. Huấn luyện **contrastive có giám sát** với [InfoNCE](https://arxiv.org/abs/1807.03748) hai chiều, hard negative và [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147).

Mô hình được đánh giá trên bốn bộ truy hồi văn bản (retrieval) và một bộ đo tương đồng ngữ nghĩa (STS) tiếng Việt công khai.

## Giới thiệu

### Động lực

Các mô hình embedding phổ biến hiện nay thường dựa trên encoder hai chiều kiểu BERT. Trong khi đó, phần lớn LLM mới là decoder-only và đã học được nhiều tri thức ngôn ngữ trong quá trình pretraining. LLM2Vec đề xuất ba bước để khai thác tri thức này cho bài toán embedding:

1. **Bật attention hai chiều**: thay causal mask bằng mask cho phép mọi token nhìn thấy nhau.
2. **MNTP**: giúp mô hình thích nghi với việc nhìn thấy ngữ cảnh ở cả hai phía.
3. **Học contrastive**: không giám sát (SimCSE) hoặc có giám sát trên cặp dữ liệu truy vấn và văn bản.

Nghiên cứu gốc thực hiện trên các LLM từ 1,3B đến 8B tham số. Dự án này kiểm tra khả năng áp dụng cùng công thức cho một mô hình nhỏ hơn khoảng 10 lần, với tokenizer chỉ dành riêng cho tiếng Việt.

### Khác biệt so với LLM2Vec gốc

- **Fine-tune toàn bộ tham số** thay vì LoRA, vì mô hình đủ nhỏ để huấn luyện trọn vẹn trên một GPU.
- **Bỏ qua bước SimCSE**: sau MNTP, mô hình được huấn luyện contrastive có giám sát ngay.
- **Matryoshka Representation Learning**: một vector 768 chiều có thể cắt ngắn mà vẫn dùng được.
- **Batch đồng nguồn và che negative giả**: tăng độ khó của in-batch negative và giảm nhiễu nhãn (xem [Giai đoạn 2](#giai-đoạn-2-contrastive-có-giám-sát)).

### Nội dung dự án

- Một pipeline hoàn chỉnh chuyển SAI_100M thành mô hình embedding, dùng lại nguyên backbone.
- Bộ đánh giá tái lập được trên năm benchmark tiếng Việt công khai, lưu theo định dạng BEIR.

## Mô hình gốc: SAI_100M

SAI_100M là mô hình ngôn ngữ decoder-only Transformer thuần Việt. Chi tiết kiến trúc và quá trình huấn luyện được mô tả tại [SAI_dev2](https://github.com/thongbuind/SAI_dev2).

| Hidden size | Attention heads | KV heads | Layers | FFN dimension | Context | Vocab |
|---:|---:|---:|---:|---:|---:|---:|
| 768 | 12 | 6 | 12 | 3.072 | 2.048 | 10.000 |

Mỗi decoder block dùng kiến trúc **pre-norm** với RMSNorm, gồm **Grouped Query Attention** có **RoPE** và feed-forward network **SwiGLU**. Không lớp tuyến tính nào dùng bias. Tokenizer là **SentencePiece Unigram** với 10.000 token, được huấn luyện trên văn bản tiếng Việt đã chuyển về chữ thường và bật byte fallback.

SAI-Embedding_100M giữ nguyên toàn bộ kiến trúc trên. Language-model head chỉ được dùng trong giai đoạn MNTP và không tham gia vào quá trình tạo embedding.

## Phương pháp

```text
SAI_100M (causal LM)
  └─ (1) Bật attention hai chiều + MNTP ──► checkpoint MNTP
       └─ (2) Contrastive có giám sát ───► SAI-Embedding_100M
```

### Attention hai chiều

Backbone cung cấp hàm `forward_features(..., causal)`. Với `causal=True`, mô hình giữ đúng hành vi của LLM gốc; chế độ này dùng để đo baseline zero-shot. Với `causal=False`, causal mask được bỏ đi, mọi token nhìn thấy nhau và chỉ các token đệm (PAD) bị chặn. Thay đổi này không thêm tham số nào. Tuy nhiên, mô hình chưa từng thấy token tương lai trong pretraining, nên cần một giai đoạn thích nghi trước khi dùng được.

### Giai đoạn 1: Masked Next-Token Prediction

MNTP kết hợp masked language modeling với mục tiêu next-token mà mô hình đã quen:

- Chọn ngẫu nhiên 20% token thật trong chuỗi (không chọn `[BOS]`). Token được chọn được thay theo tỉ lệ 80/10/10: 80% thay bằng token mask, 10% thay bằng token ngẫu nhiên, 10% giữ nguyên.
- Token bị mask ở vị trí *i* được dự đoán từ hidden state ở vị trí *i − 1*, qua language-model head có sẵn. Cách dự đoán này giữ nguyên quan hệ giữa hidden state và token mà mô hình đã học khi pretraining.
- Tokenizer của SAI không có token `[MASK]`, nên dự án dùng `[UNK]` thay thế, tương tự LLM2Vec dùng `_` cho Llama.

Giai đoạn này chỉ giúp mô hình quen với ngữ cảnh hai chiều, chưa dạy ngữ nghĩa câu, nên chỉ cần văn bản thuần và vài nghìn bước huấn luyện.

### Giai đoạn 2: Contrastive có giám sát

**Định dạng đầu vào.** Văn bản được chuẩn hoá (NFC, chữ thường, gộp khoảng trắng) cho khớp với dữ liệu tokenizer đã học. Nếu bỏ bước này, cùng một câu ở dạng NFD bị tách thành số token gấp khoảng 4 lần. Tương tự E5, mô hình dùng prefix để phân biệt vai trò của văn bản:

- `truy vấn: ` cho câu hỏi.
- `văn bản: ` cho tài liệu.
- Với task đối xứng (NLI, STS), cả hai vế đều dùng `truy vấn: `.

**Pooling.** Embedding của câu là trung bình hidden state của các token không phải PAD, sau đó được L2-normalize.

**Hàm loss.** Mô hình dùng InfoNCE hai chiều trên in-batch negative và hard negative. Với batch gồm $B$ cặp $(q_i, p_i)$ và tập passage $\mathcal{P}$ gồm $B$ positive cùng toàn bộ hard negative của batch:

```math
\mathcal{L}_{q \to p} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\cos(q_i, p_i)/\tau)}{\sum_{p_j \in \mathcal{P} \setminus \mathcal{F}_i} \exp(\cos(q_i, p_j)/\tau)}, \qquad \mathcal{L} = \tfrac{1}{2}\left(\mathcal{L}_{q \to p} + \mathcal{L}_{p \to q}\right)
```

Trong đó $\mathcal{F}_i$ là tập **negative giả** của mẫu $i$: các passage trùng nội dung với positive $p_i$, hoặc là positive của một dòng khác có cùng truy vấn. Chiều $p \to q$ được tính tương tự trên $B$ truy vấn.

**Matryoshka.** Loss được tính trên các prefix 768 / 512 / 256 / 128 / 64 chiều của vector, mỗi prefix được L2-normalize lại trước khi tính. Loss tổng là trung bình có trọng số của các loss thành phần. Nhờ đó, người dùng có thể cắt ngắn embedding để giảm chi phí lưu trữ và tìm kiếm.

**Batch đồng nguồn.** Mỗi batch chỉ lấy mẫu từ một nguồn dữ liệu. Khi đó in-batch negative cùng lĩnh vực với positive (luật với luật, tin tức với tin tức) nên khó phân biệt hơn, và task đối xứng không bị trộn với task bất đối xứng trong cùng một loss.

**GradCache.** Kỹ thuật GradCache (Gao et al., 2021) tách batch thành các chunk nhỏ khi forward và backward. Nhờ đó batch 512 cặp chạy được trên GPU 24GB, trong khi gradient vẫn tương đương khi tính trên cả batch một lần.

**Chọn checkpoint.** Checkpoint cuối cùng được chọn theo điểm trên bộ dev, hoàn toàn tách biệt với tập đánh giá.

### Siêu tham số

| | Giai đoạn 1: MNTP | Giai đoạn 2: Contrastive |
|---|---|---|
| Khởi tạo | SAI_100M | Checkpoint sau MNTP |
| Số bước | 3.000 | 1 epoch |
| Batch | 64 chuỗi × 512 token | 512 cặp + tối đa 3 hard negative/mẫu (GradCache, chunk 128) |
| Độ dài tối đa | 512 | Query 64, passage 256 (512 với nguồn văn bản dài) |
| Optimizer | AdamW, weight decay 0,01 | AdamW, weight decay 0,01 |
| Learning rate | 5e-5, warmup 200 bước | 5e-5, warmup 125 bước |
| Khác | Mask 20% (80/10/10) | τ = 0,02; trọng số Matryoshka 1 / 1 / 0,5 / 0,5 / 0,25; đánh giá dev mỗi 200 bước |

Cả hai giai đoạn dùng cùng một lịch learning rate:

1. Warmup tuyến tính trong số bước đã cho.
2. Giữ nguyên learning rate đến 40% tổng số bước.
3. Giảm tuyến tính về 10% giá trị ban đầu tại mốc 70%, sau đó giữ nguyên.

Gradient được clip ở norm 1,0. Trên GPU, mô hình huấn luyện với mixed precision bf16.

## Thiết lập đánh giá

| Task | Lĩnh vực | Nguồn | Quy mô |
|---|---|---|---|
| Zalo Legal | Luật | [`GreenNode/zalo-ai-legal-text-retrieval-vn`](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) (test) | 788 query / 61k văn bản |
| TVPL | Luật | [`GreenNode/TVPL-Retrieval-VN`](https://huggingface.co/datasets/GreenNode/TVPL-Retrieval-VN) (test) | 3.000 query / 10,6k văn bản |
| ViQuAD | Wikipedia | [`taidng/UIT-ViQuAD2.0`](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0) (test) | 7.184 query / 1.241 đoạn |
| nano-MSMARCO-vi | Web | [`GreenNode/nano-msmarco-vn`](https://huggingface.co/datasets/GreenNode/nano-msmarco-vn) (dev) | 3.498 query / 104k đoạn |
| STS-B-vi | Tương đồng câu | [`GreenNode/stsbenchmark-sts-vn`](https://huggingface.co/datasets/GreenNode/stsbenchmark-sts-vn) (test) | 1.379 cặp |

- **Độ đo:** nDCG@10, MRR@10, Recall@10 và Recall@100 cho retrieval; hệ số tương quan Spearman cho STS.
- **Điểm tổng hợp (main score):** trung bình nDCG@10 của các task retrieval và Spearman của STS, tính ở 768 chiều.
- **Chống rò rỉ dữ liệu:** chỉ dùng split test (riêng nano-MSMARCO dùng split dev). Mọi truy vấn trong dữ liệu huấn luyện trùng với truy vấn của tập dev hoặc eval đều bị loại bỏ.
- Khi đo các mô hình khác, prefix và cách pooling được đặt đúng theo model card của từng mô hình.

## Kết quả thực nghiệm

### So sánh với các mô hình khác

Bảng dưới báo cáo nDCG@10 cho các task retrieval và Spearman cho STS-B-vi. Mỗi mô hình được đo ở số chiều đầy đủ của nó. Main score là trung bình của năm cột. Giá trị cao nhất mỗi cột được in đậm.

| Mô hình | Tham số | Chiều | Zalo Legal | TVPL | ViQuAD | nano-MSMARCO | STS-B-vi | Main score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **SAI-Embedding_100M** | 114M | 768 | 71,10 | 79,15 | 76,25 | 74,84 | 75,56 | 75,38 |
| `bkai-foundation-models/vietnamese-bi-encoder` (base PhoBERT) | 135M | 768 | **84,55** | 77,17 | 70,62 | 75,34 | 73,40 | 76,21 |
| `BAAI/bge-m3` (500M) | 568M | 1.024 | 71,35 | **81,21** | **82,54** | **83,76** | **81,15** | **80,00** |

So với `vietnamese-bi-encoder` có cỡ tương đương, SAI-Embedding_100M có main score thấp hơn 0,83 điểm. SAI-Embedding_100M cao hơn trên TVPL (+1,98), ViQuAD (+5,63) và STS-B-vi (+2,16), nhưng thấp hơn rõ trên Zalo Legal (−13,45) và thấp hơn nhẹ trên nano-MSMARCO (−0,50). `bge-m3` lớn hơn khoảng 5 lần và dẫn đầu ở bốn trên năm task. Trên Zalo Legal, SAI-Embedding_100M đạt điểm gần bằng `bge-m3` (71,10 so với 71,35).

## Tải mô hình và sử dụng

Tải toàn bộ repository (gồm checkpoint, tokenizer, cấu hình và mã nguồn), sau đó chạy ví dụ từ thư mục vừa tải:

```bash
hf download thongbuind/SAI-Embedding_100M --local-dir SAI-Embedding_100M
cd SAI-Embedding_100M
pip install -r requirements-inference.txt
```

Đường chạy inference dùng các lớp và hàm tự viết trong repository: `EmbeddingModel`, `TransformerModel`, `EmbeddingTokenizer`, chuẩn hoá văn bản, pooling và `encode()`. Không dùng `AutoModel`, `SentenceTransformer`, `pipeline()` hoặc mô hình embedding đóng gói sẵn. PyTorch cung cấp phép toán tensor và các toán tử nền; SentencePiece đọc tokenizer gốc. Lệnh `hf download` chỉ tải file, không tải hay khởi tạo kiến trúc mô hình thay cho mã nguồn của dự án.

Ví dụ tìm văn bản liên quan tới một câu hỏi:

```python
from src.data.embedding_data import EmbeddingTokenizer
from src.model.EmbeddingModel import EmbeddingModel

tokenizer = EmbeddingTokenizer()
model = EmbeddingModel.from_checkpoint("model/SAI-Embedding_100M.pt").eval()

queries = model.encode(
    ["người đi bộ vượt đèn đỏ bị phạt bao nhiêu"],
    tokenizer,
    is_query=True,
)
docs = model.encode(
    ["Điều 9. Xử phạt người đi bộ vi phạm quy tắc giao thông ..."],
    tokenizer,
    is_query=False,
)
scores = queries @ docs.T  # cosine similarity
print(scores)
```

`encode()` tự thêm prefix theo `is_query`, chuẩn hoá văn bản và trả về tensor `float32` đã L2-normalize. Truyền thêm `dim=256` (hoặc 512 / 128 / 64) để lấy embedding Matryoshka ngắn hơn. Với task đối xứng như so sánh hai câu, dùng `is_query=True` cho cả hai vế.

## Hạn chế

- **Không phân biệt chữ hoa, chữ thường.** Tokenizer của SAI được huấn luyện trên văn bản chữ thường, nên mọi đầu vào đều bị chuyển về chữ thường. Thông tin viết hoa (tên riêng, viết tắt) bị mất.
- **Chỉ dành cho tiếng Việt.** Vocabulary 10.000 token được tối ưu cho tiếng Việt. Ngôn ngữ khác chủ yếu được mã hoá qua byte fallback, nên chất lượng embedding không được đảm bảo.
- **Văn bản dài.** Mô hình được huấn luyện với passage 256–512 token. Văn bản dài hơn nên được chia đoạn trước khi encode.
- **Khả năng tái lập.** Repo phát hành mô hình để sử dụng và đánh giá. Dữ liệu và cấu hình huấn luyện không được công bố, nên chỉ phần đánh giá có thể tái lập.

## Tài liệu tham khảo

- BehnamGhader, P., Adlakha, V., Mosbach, M., Bahdanau, D., Chapados, N., & Reddy, S. (2024). *LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders*. [arXiv:2404.05961](https://arxiv.org/abs/2404.05961)
- Gao, L., Zhang, Y., Han, J., & Callan, J. (2021). *Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup*. [arXiv:2101.06983](https://arxiv.org/abs/2101.06983)
- Kusupati, A., et al. (2022). *Matryoshka Representation Learning*. [arXiv:2205.13147](https://arxiv.org/abs/2205.13147)
- van den Oord, A., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding*. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
- Wang, L., et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. [arXiv:2212.03533](https://arxiv.org/abs/2212.03533)
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)
