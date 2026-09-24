import os
import random
import json
from google import genai
from dotenv import load_dotenv

"""
Generator Agent
- LLM 名稱：Gen_LLM
- 功能：根據 CS 主題產生原始 MCQ 題目（含答案與解析）
"""

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEN_LLM_NAME = "Gen_LLM"
GEN_MODEL_NAME = "gemini-3-pro-preview"

QUESTION_PROMPT = """
你是一位資工領域的出題專家 (Gen_LLM)。
請根據主題編寫一道單選題（四選一），並提供題目與四個選項、答案與解析。
使用繁體中文。

格式：
題目：xxxx
(A) xxx
(B) xxx
(C) xxx
(D) xxx
答案：X
解析：xxxx
"""



# 人類語氣的 prompt 範本，供隨機挑選以產生多樣化的 QUESTION_PROMPT。
# 這些範本模仿自然講話方式（Alpaca-style augmentation）。
PROMPT_TEMPLATES = [
    "請幫我出一題有關 {topic} 的單選題(四個選項)",
    "你是一位大學的資工老師，幫我針對 {topic} 出一個考試用的單選題（四選一）",
    "我想要一題關於 {topic} 的單選題(四個選項)。",
    "以考古題風格出一題關於 {topic} 的單選題。",
    "請幫我設計一道有關 {topic} 的單選題（四個選項）。",
    "幫我生成一題選擇題，主題是 {topic}。",
]


def build_question_prompt(topic: str) -> str:
    """
    從 `PROMPT_TEMPLATES` 隨機選一個範本並以 topic 格式化，回傳最終要送給 LLM 的 prompt。
    這會用來模擬多樣且自然的人類提問方式（類似 Alpaca 的多樣化 seed）。
    """
    t = random.choice(PROMPT_TEMPLATES).strip()
    return t.format(topic=topic)


# 嘗試載入外部擴增後的 prompts.json（若使用 instruction_generator.py 產生）
try:
    prompts_path = os.path.join(os.path.dirname(__file__), "prompts.json")
    if os.path.exists(prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as _f:
            data = json.load(_f)
            if isinstance(data, list) and data:
                PROMPT_TEMPLATES = data
except Exception:
    # 若載入失敗，不影響內建範本
    pass

CS_TOPICS = [
    # ===== 資料結構 Data Structures =====
    "資料結構 - 陣列(Array)",
    "資料結構 - 動態陣列(Dynamic Array)",
    "資料結構 - 連結串列(Linked List)",
    "資料結構 - 堆疊(Stack)",
    "資料結構 - 佇列(Queue)",
    "資料結構 - 雙端佇列(Deque)",
    "資料結構 - 雜湊表(Hash Table)",
    "資料結構 - 集合(Set)",
    "資料結構 - 位元集合(Bitset)",
    "資料結構 - 字串(String)",
    "資料結構 - Trie/Prefix Tree",
    "資料結構 - Heap / Priority Queue",
    "資料結構 - 二元搜尋樹(Binary Search Tree, BST)",
    "資料結構 - AVL Tree",
    "資料結構 - Red-Black Tree",
    "資料結構 - B-Tree / B+ Tree",
    "資料結構 - Segment Tree",
    "資料結構 - Fenwick Tree (Binary Indexed Tree, BIT)",
    "資料結構 - Union-Find (Disjoint Set Union, DSU)",
    "資料結構 - Skip List",
    "資料結構 - Bloom Filter",
    "資料結構 - LRU Cache",
    "資料結構 - 圖(Graph) 表示法（Adjacency List/Matrix）",

    # ===== 演算法 Algorithms =====
    "演算法 - 漸進複雜度(Big-O) 與分析(Analysis of Algorithms)",
    "演算法 - 遞迴(Recursion) 與分治(Divide and Conquer)",
    "演算法 - 排序(Sorting)：QuickSort / MergeSort / HeapSort",
    "演算法 - 搜尋(Searching)：Binary Search",
    "演算法 - 雙指標(Two Pointers) 與滑動視窗(Sliding Window)",
    "演算法 - 回溯(Backtracking)",
    "演算法 - 動態規劃(Dynamic Programming, DP)",
    "演算法 - 貪婪(Greedy)",
    "演算法 - 圖論(Graph Algorithms)：BFS / DFS",
    "演算法 - 最短路徑(Shortest Path)：Dijkstra / Bellman-Ford",
    "演算法 - 全點對最短路徑(All-Pairs)：Floyd-Warshall",
    "演算法 - 最小生成樹(MST)：Kruskal / Prim",
    "演算法 - 拓樸排序(Topological Sort)",
    "演算法 - 強連通分量(SCC)：Kosaraju / Tarjan",
    "演算法 - 最大流/最小割(Max Flow/Min Cut)：Edmonds-Karp / Dinic",
    "演算法 - 匹配(Matching)：Bipartite Matching / Hungarian",
    "演算法 - 字串比對(String Matching)：KMP / Rabin-Karp / Z-algorithm",
    "演算法 - 近似演算法(Approximation Algorithms)",
    "演算法 - 隨機化演算法(Randomized Algorithms)",
    "演算法 - NP-Complete 與計算複雜度(Computational Complexity)",
    "演算法 - 平行演算法(Parallel Algorithms)",

    # ===== 程式設計與軟體工程 Programming & SE =====
    "程式設計 - 物件導向(Object-Oriented Programming, OOP)",
    "程式設計 - 函數式程式設計(Functional Programming)",
    "程式設計 - 記憶體管理(Memory Management)",
    "程式設計 - 指標與參照(Pointers & References)",
    "程式設計 - 例外處理(Exception Handling)",
    "程式設計 - 測試(Test)：Unit Test / Integration Test",
    "軟體工程 - 版本控制(Git) 與分支策略(Branching Strategy)",
    "軟體工程 - 設計模式(Design Patterns)",
    "軟體工程 - SOLID 原則(SOLID Principles)",
    "軟體工程 - 重構(Refactoring)",
    "軟體工程 - CI/CD",
    "軟體工程 - 微服務(Microservices)",
    "軟體工程 - API 設計(REST / GraphQL)",
    "軟體工程 - 可觀測性(Observability)：Logging / Metrics / Tracing",
    "軟體工程 - 效能分析(Profiling) 與最佳化(Optimization)",

    # ===== 作業系統 Operating Systems =====
    "作業系統 - 行程/執行緒(Process/Thread)",
    "作業系統 - CPU 排程(CPU Scheduling)",
    "作業系統 - 同步(Synchronization)：Mutex / Semaphore / Monitor",
    "作業系統 - 競態條件(Race Condition)",
    "作業系統 - 死結(Deadlock)",
    "作業系統 - 記憶體管理(Virtual Memory / Paging)",
    "作業系統 - 置換演算法(Page Replacement)：LRU / FIFO",
    "作業系統 - 檔案系統(File System)：Inode / Journaling",
    "作業系統 - 系統呼叫(System Calls)",
    "作業系統 - I/O 與中斷(I/O & Interrupts)",
    "作業系統 - 容器與命名空間(Containers & Namespaces)",
    "作業系統 - 虛擬化(Virtualization)：Hypervisor",

    # ===== 計算機網路 Computer Networks =====
    "計算機網路 - OSI/TCP-IP 模型(OSI/TCP-IP Model)",
    "計算機網路 - Ethernet 與 MAC(Ethernet/MAC)",
    "計算機網路 - IP / Subnetting (CIDR)",
    "計算機網路 - ARP / DHCP / DNS",
    "計算機網路 - TCP：三向交握(3-way handshake) 與流量控制",
    "計算機網路 - TCP 擁塞控制(Congestion Control)",
    "計算機網路 - UDP 與 QUIC",
    "計算機網路 - Routing：OSPF / BGP",
    "計算機網路 - NAT / Firewall",
    "計算機網路 - TLS/HTTPS",
    "計算機網路 - HTTP/1.1 vs HTTP/2 vs HTTP/3",
    "計算機網路 - WebSocket",
    "計算機網路 - CDN 與快取(Cache)",
    "計算機網路 - 網路安全概念(Network Security Basics)",

    # ===== 資料庫 Databases =====
    "資料庫 - 關聯式模型(Relational Model)",
    "資料庫 - SQL：Join / Group By / Window Function",
    "資料庫 - 正規化(Normalization)",
    "資料庫 - 索引(Index)：B+ Tree / Hash Index",
    "資料庫 - 查詢最佳化(Query Optimization)",
    "資料庫 - 交易(Transaction) 與 ACID",
    "資料庫 - 隔離等級(Isolation Levels)",
    "資料庫 - 併發控制(Concurrency Control)：2PL / MVCC",
    "資料庫 - 復原與日誌(Recovery & WAL)",
    "資料庫 - 分割與分片(Partitioning & Sharding)",
    "資料庫 - 複寫(Replication) 與一致性(Consistency)",
    "資料庫 - NoSQL：Key-Value / Document / Column / Graph",
    "資料庫 - 時序資料庫(Time-Series DB)",
    "資料庫 - 資料倉儲(Data Warehouse) 與 OLAP",
    "資料庫 - 事件串流(Event Streaming)：Kafka 基礎",

    # ===== 計算機架構與硬體 Computer Architecture =====
    "計算機架構 - 指令集架構(ISA)：RISC vs CISC",
    "計算機架構 - Pipeline 與 Hazard",
    "計算機架構 - 快取(Cache) 與記憶體階層(Memory Hierarchy)",
    "計算機架構 - 虛擬記憶體(Virtual Memory) 與 TLB",
    "計算機架構 - 分支預測(Branch Prediction)",
    "計算機架構 - 亂序執行(Out-of-Order Execution)",
    "計算機架構 - SIMD / 向量化(Vectorization)",
    "計算機架構 - GPU 架構(GPU Architecture)",
    "計算機架構 - 多核心與一致性(Cache Coherence)",
    "計算機架構 - 匯流排與 I/O(Bus & I/O)",

    # ===== 編譯器與程式語言 Compilers & PL =====
    "編譯器 - 詞法分析(Lexical Analysis)",
    "編譯器 - 語法分析(Parsing)：LL/LR",
    "編譯器 - 抽象語法樹(Abstract Syntax Tree, AST)",
    "編譯器 - 中介表示(Intermediate Representation, IR)",
    "編譯器 - 最佳化(Compiler Optimizations)",
    "程式語言 - 型別系統(Type Systems)：Static/Dynamic",
    "程式語言 - 記憶體安全(Memory Safety)",
    "程式語言 - Garbage Collection (GC)",
    "程式語言 - Concurrency Model（Actor / CSP）",

    # ===== 分散式系統 Distributed Systems =====
    "分散式系統 - CAP 定理(CAP Theorem)",
    "分散式系統 - 一致性模型(Consistency Models)",
    "分散式系統 - 共識(Consensus)：Paxos / Raft",
    "分散式系統 - 分散式鎖(Distributed Lock)",
    "分散式系統 - 時鐘與排序(Clock)：Lamport / Vector Clock",
    "分散式系統 - Leader Election",
    "分散式系統 - 容錯(Fault Tolerance)",
    "分散式系統 - 兩階段提交(2PC) 與三階段提交(3PC)",
    "分散式系統 - MapReduce 與資料處理",
    "分散式系統 - Stream Processing（Flink / Spark Streaming）",
    "分散式系統 - Service Discovery",

    # ===== 資訊安全 Cybersecurity =====
    "資訊安全 - 密碼學基礎(Cryptography Basics)",
    "資訊安全 - 對稱/非對稱加密(Symmetric/Asymmetric Encryption)",
    "資訊安全 - 雜湊(Hash) 與 HMAC",
    "資訊安全 - 數位簽章(Digital Signature)",
    "資訊安全 - 金鑰交換(Key Exchange)：Diffie-Hellman",
    "資訊安全 - 身分驗證(Authentication) 與授權(Authorization)",
    "資訊安全 - OAuth2 / OpenID Connect",
    "資訊安全 - Web 安全：XSS / CSRF / SQL Injection",
    "資訊安全 - 安全開發(SDL) 與威脅建模(Threat Modeling)",
    "資訊安全 - 漏洞與攻擊面(Vulnerabilities & Attack Surface)",
    "資訊安全 - 惡意程式(Malware) 基礎",
    "資訊安全 - 緩衝區溢位(Buffer Overflow) 概念",

    # ===== 人工智慧與機器學習 AI & ML =====
    "機器學習 - 監督/非監督/強化學習(Supervised/Unsupervised/RL)",
    "機器學習 - 線性模型(Linear Models)：Linear/Logistic Regression",
    "機器學習 - 決策樹與集成(Tree Ensembles)：Random Forest / XGBoost",
    "機器學習 - 神經網路(Neural Networks)",
    "深度學習 - 反向傳播(Backpropagation)",
    "深度學習 - CNN / RNN / Transformer",
    "深度學習 - 正規化與泛化(Regularization & Generalization)",
    "深度學習 - 最佳化(Optimization)：SGD / Adam",
    "深度學習 - 表徵學習(Representation Learning)",
    "自然語言處理 - 語言模型(Language Models) 與 Tokenization",
    "自然語言處理 - 指令微調(Instruction Tuning)",
    "電腦視覺 - 影像分類(Image Classification)",
    "電腦視覺 - 物件偵測(Object Detection)",
    "電腦視覺 - 分割(Segmentation)",
    "強化學習 - MDP 與策略梯度(Policy Gradient)",
    "資料探勘 - 分群(Clustering) 與降維(Dimensionality Reduction)",

    # ===== 資料科學與工程 Data Science & Engineering =====
    "資料工程 - ETL/ELT",
    "資料工程 - 資料品質(Data Quality)",
    "資料工程 - 特徵工程(Feature Engineering)",
    "資料工程 - 大數據(Big Data)：Spark 基礎",
    "資料工程 - 資料湖(Data Lake) 與 Lakehouse",
    "資料工程 - 資料治理(Data Governance)",
    "MLOps - 模型部署(Model Deployment)",
    "MLOps - 模型監控(Model Monitoring) 與漂移(Drift)",
    "MLOps - 實驗追蹤(Experiment Tracking)：MLflow",

    # ===== 理論計算機科學 Theory =====
    "理論 - 正則語言與有限自動機(Regular Languages & DFA/NFA)",
    "理論 - 上下文無關文法(CFG) 與 Pushdown Automata",
    "理論 - 圖靈機(Turing Machine) 與可計算性(Computability)",
    "理論 - P/NP 與 NP-Complete",
    "理論 - 近似與隨機(Approximation & Randomization)",
    "理論 - 資訊理論(Information Theory)：Entropy",

    # ===== 人機互動與圖形 HCI & Graphics =====
    "人機互動 - 可用性(Usability) 與使用者研究(User Study)",
    "人機互動 - 互動設計(Interaction Design)",
    "電腦圖學 - 2D/3D 轉換(Transforms)",
    "電腦圖學 - 光照模型(Lighting Models)",
    "電腦圖學 - 光柵化(Rasterization)",
    "電腦圖學 - Ray Tracing",
    "遊戲開發 - 遊戲迴圈(Game Loop) 與物理引擎(Physics Engine)",

    # ===== 嵌入式與 IoT Embedded & IoT =====
    "嵌入式系統 - MCU 與中斷(Interrupts)",
    "嵌入式系統 - 即時系統(Real-Time Systems, RTOS)",
    "嵌入式系統 - 通訊協定(I2C / SPI / UART)",
    "IoT - 感測器(Sensors) 與資料收集",
    "IoT - MQTT / CoAP",

    # ===== 雲端與 DevOps Cloud & DevOps =====
    "雲端 - 虛擬機(VM) 與容器(Container)",
    "雲端 - Kubernetes 基礎",
    "雲端 - Serverless",
    "雲端 - 負載平衡(Load Balancing)",
    "雲端 - 快取與訊息佇列(Cache & Message Queue)",
    "DevOps - IaC (Infrastructure as Code)：Terraform",
    "DevOps - SRE 基礎：SLI/SLO/Error Budget",

    # ===== 其他常見 CS 子領域 Others =====
    "軟體安全 - 靜態分析(Static Analysis)",
    "軟體安全 - 動態分析(Dynamic Analysis)",
    "資訊檢索 - 搜尋引擎(Information Retrieval)",
    "推薦系統 - 協同過濾(Collaborative Filtering)",
    "平行與高效能運算 - 多執行緒(Threads) 與 OpenMP",
    "平行與高效能運算 - GPU 計算(CUDA) 基礎",
    "形式化方法 - 模型檢查(Model Checking)",
]



from GeminiAgent.agent.llm_utils import generate_content_with_tokens


def call_gen_llm(prompt: str) -> tuple:
    """Call generator model and return (text, out_tokens, in_tokens)."""
    text, out_tokens, in_tokens = generate_content_with_tokens(GEN_MODEL_NAME, prompt)
    return text, out_tokens, in_tokens


def augment_prompt_templates(target_total: int = 500, per_template: int = 4) -> list:
    """
    使用 LLM 自動擴增 `PROMPT_TEMPLATES`（Alpaca-style self-instruct）。

    不只是改寫現有範本，而是讓 LLM 基於種子的**精神與風格**產生完全新的提問方式，
    鼓勵多樣性與創意。

    - `target_total`: 擴增後希望的總數量（包含原始範本）。
    - `per_template`: 每個原始範本最多嘗試產生的創意變體數。

    回傳擴增後的範本清單（list of str）。

    注意：此方法會呼叫 `call_gen_llm`，需設定好 `GEMINI_API_KEY`。
    """
    seeds = PROMPT_TEMPLATES.copy()
    # 快速檢查：如果已經夠多，直接回傳
    if len(seeds) >= target_total:
        return seeds

    # 組裝提示給 LLM：鼓勵自由創意產生新風格（不限於改寫現有範本）
    seed_block = "\n".join([f"- {s}" for s in seeds])
    prompt_template = (
        "你是一位創意提示語（prompt）設計專家，使用繁體中文。\n\n"
        "下列是一些「出題與教學」相關的提示語範例：\n"
        f"{seed_block}\n\n"
        "請根據上述範例所代表的**精神**與**多樣性**，產生 {per_template} 個全新且有創意的提示語。\n"
        "這些新提示語應該：\n"
        "1. 包含不同的提問情境（考試、練習、複習、教學等）\n"
        "2. 使用不同的語氣與表達方式（正式、親切、簡潔、詳細等）\n"
        "3. 保留佔位符 `{topic}`（用於後續代入具體主題）\n"
        "4. 避免與原始範例過於相似\n\n"
        "請輸出一個 JSON 陣列，內容為字串清單（每個字串為一個新提示語範本）。\n"
        "注意：不要替換 `{topic}` 佔位符；該佔位符必須保留以便後續使用。"
    )

    # 使用 replace 來替換 {per_template}，避免同時解析到 `{topic}` 造成 KeyError
    prompt = prompt_template.replace("{per_template}", str(per_template))

    resp_text, resp_out_tokens, resp_in_tokens = call_gen_llm(prompt)
    candidates = []
    if resp_text:
        # 嘗試解析 JSON
        try:
            j = json.loads(resp_text)
            if isinstance(j, list):
                candidates = [s.strip() for s in j if isinstance(s, str) and s.strip()]
        except Exception:
            # 如果不是 JSON，嘗試依行拆分（較寬鬆的後備）
            for line in resp_text.splitlines():
                s = line.strip().lstrip("- ").lstrip("* ").strip('\"\'')
                if s and len(s) > 5:  # 過濾太短的行（可能是雜訊）
                    candidates.append(s)

    # 去重並保留原本的 seeds
    seen = set(seeds)
    augmented = seeds.copy()
    valid_count = 0
    
    for c in candidates:
        if len(augmented) >= target_total:
            break
        
        # 必須包含 {topic} placeholder
        if "{topic}" not in c:
            continue
        
        # 避免重複
        if c in seen:
            continue
        
        # 加入新範本
        augmented.append(c)
        seen.add(c)
        valid_count += 1

    print(f"[augment_prompt_templates] 成功新增 {valid_count} 個新範本，總計 {len(augmented)} 個")
    return augmented


def save_prompt_templates(path: str, templates: list):
    """將範本清單存成 JSON 檔案（UTF-8）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 簡單的 CLI：執行此檔會用 LLM 擴增並儲存到 agent/prompts.json
    out_path = os.path.join(os.path.dirname(__file__), "prompts.json")
    print("Start augmenting prompt templates via Gemini...\nThis will call the API (make sure GEMINI_API_KEY is set).")
    new_templates = augment_prompt_templates(target_total=500, per_template=4)
    save_prompt_templates(out_path, new_templates)
    print(f"Saved {len(new_templates)} templates to {out_path}")


def generate_raw_question(topic: str) -> tuple:
    """
    使用 Gen_LLM 依據 topic 產生原始題目（含答案與解析）。

    回傳 (used_prompt, model_output)
    - used_prompt: 實際送給 LLM 的人類語氣 prompt（可存入 dataset 的 question 欄位）
    - model_output: LLM 回傳的題目內容
    """
    # 先建立一個人類語氣的 prompt（多樣化）
    human_prompt = build_question_prompt(topic)

    # 將我們的格式化 QUESTION_PROMPT 與人類 prompt 合併，讓 LLM 既知道要的格式，也感受到自然口語的表述
    msgs = f"{human_prompt}\n\n格式要求：\n{QUESTION_PROMPT}\n請根據主題「{topic}」出一題。"

    out_text, out_tokens, in_tokens = call_gen_llm(msgs)

    if not out_text:
        out_text = f"題目（Gen_LLM 回傳空白）請出與 {topic} 有關的題目。"

    return human_prompt, out_text, out_tokens, in_tokens


def random_topic() -> str:
    return random.choice(CS_TOPICS)