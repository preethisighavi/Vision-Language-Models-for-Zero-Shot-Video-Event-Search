# System Architecture & Design: Zero-Shot Video & Event Search

---

## 1. Overall System Architecture (Workbook 5.2 — System Design)

```mermaid
flowchart TB
    %% Actors
    EndUser((End User))
    DataEng((Data Engineer))

    %% Frontend
    subgraph Frontend [Presentation Layer — Web UI]
        SearchBar[Natural Language Search Bar]
        VideoPlayer[Video Player + Timestamp Slider]
        Dashboard[Analytics Dashboard]
    end

    %% Backend API
    subgraph Backend [Application Layer — REST API]
        SearchAPI[FastAPI Search Gateway]
        TextEncoder[VLM Text Encoder\nCLIP / BLIP / Flamingo]
        Ranker[Score Fusion & Ranker]
    end

    %% Data Engineering Pipeline
    subgraph Pipeline [Data Engineering Layer — vlm_pipeline]
        Orchestrator[Batch Orchestrator\nrun_batches.py]
        Ingest[Dataset Ingestion\ningest.py — MSR-VTT & MSVD]
        Splits[Split Generator\nbuild_splits.py — zero-leakage]
        Keyframe[Keyframe Extractor\nkeyframe_extract.py — PyAV + PIL]
        VisionEncoder[VLM Vision Encoder Batch\nclip / blip / flamingo]
        Export[Result Exporter\nexport_results.py]
    end

    %% Storage
    subgraph Storage [Data Storage & Infrastructure]
        RawStorage[(Object Storage\nRaw .mp4 + JSON)]
        Parquet[(Apache Parquet\ntrain/val/test splits)]
        VectorDB[(Vector Database\nPinecone / Chroma)]
    end

    %% Online Search Flow
    EndUser -- "1. Types natural language query" --> SearchBar
    SearchBar -- "2. GET /search?q=..." --> SearchAPI
    SearchAPI -- "3. Encode text" --> TextEncoder
    TextEncoder -- "4. Query vector" --> Ranker
    Ranker -- "5. Cosine similarity search" --> VectorDB
    VectorDB -. "6. Top-K (video_id, timestamp, score)" .-> Ranker
    Ranker -- "7. Ranked JSON results" --> SearchAPI
    SearchAPI -- "8. Return payload" --> VideoPlayer
    VideoPlayer -- "9. Seek to localized timestamp" --> EndUser

    %% Offline Pipeline Flow
    DataEng -- "Schedule batch" --> Orchestrator
    Orchestrator --> Ingest
    RawStorage -- "Stream videos + annotations" --> Ingest
    Ingest --> Splits
    Splits --> Parquet
    Splits --> Keyframe
    Keyframe -- "224x224 center-cropped JPEGs" --> VisionEncoder
    VisionEncoder -- "Frame embeddings" --> VectorDB
    VisionEncoder --> Export
```

---

## 2. Machine Learning Data Flow (Workbook 4.2 — Model Supports)

```mermaid
flowchart LR
    subgraph Input [Input Data]
        V[Raw Video .mp4]
        J[JSON Caption Annotations]
    end

    subgraph Preprocessing [Preprocessing Stage]
        KE[Keyframe Extraction\nPyAV — 3 equidistant frames]
        CC[Center Crop 224x224\nAspect-ratio preserved]
        TOK[Caption Tokenizer\nCLIP / BLIP Tokenizer max=77]
    end

    subgraph Models [VLM Encoding Stage]
        CLIP[CLIP ViT-B/32\nImage Encoder → 512-dim vector]
        BLIP[BLIP Image Encoder\n→ 1024-dim vector]
        FLAM[Flamingo-Lite\nTemporal Frame Sequence → 512-dim]
        SPART[Spartan-VLM\nCLIP backbone + LSTM → 256-dim]
        MKD[MKD Student\nDistilled ViT-L → 512-dim]
    end

    subgraph Index [Indexing Stage]
        VDB[(Pinecone / Chroma\nVector Index)]
    end

    subgraph QueryFlow [Query Encoding]
        TQ[User Text Query]
        TE[Text Encoder\nCLIP / BLIP Text Transformer]
        QV[Query Vector]
        SR[Similarity Search\nCosine Distance]
    end

    V --> KE --> CC
    J --> TOK

    CC --> CLIP --> VDB
    CC --> BLIP --> VDB
    CC --> FLAM --> VDB
    CC --> SPART --> VDB
    CC --> MKD --> VDB

    TQ --> TE --> QV --> SR
    VDB -. "Top-K matches" .-> SR
```

---

## 3. Use Case Diagram (Workbook 5.1 — System Requirements)

```mermaid
flowchart TD
    EU((End User))
    DE((Data Engineer))
    MR((Model Researcher))

    EU --> UC1[Search videos by natural language]
    EU --> UC2[View video at localized timestamp]
    EU --> UC3[Browse analytics dashboard]

    DE --> UC4[Ingest new dataset batch]
    DE --> UC5[Run keyframe extraction pipeline]
    DE --> UC6[Refresh vector index]

    MR --> UC7[Evaluate model Recall-at-K]
    MR --> UC8[Swap/add new VLM model]
    MR --> UC9[Fine-tune Spartan-VLM]

    UC1 --> S[System Boundary]
    UC2 --> S
    UC3 --> S
    UC4 --> S
    UC5 --> S
    UC6 --> S
    UC7 --> S
    UC8 --> S
    UC9 --> S
```

---

## 4. Data Pipeline Detail (Demo — Data Pre-Processing & Training/Test Splits)

```mermaid
flowchart TD
    A[Raw MSR-VTT\n10,000 videos] --> C[ingest.py]
    B[Raw MSVD\n~2,000 videos] --> C
    C --> D[quality_filter.py\nDrop corrupt / zero-length videos]
    D --> E[build_splits.py\nStratified by category]
    E --> F[split_train.parquet\n~75 percent]
    E --> G[split_val.parquet\n~10 percent]
    E --> H[split_test.parquet\n~15 percent]
    F --> I[keyframe_extract.py\n3 frames per video\n224x224 center crop]
    G --> I
    H --> I
    I --> J[with_frames.parquet\nframe_paths column added]
    J --> K[VLM Vision Encoder\nBatch inference]
    K --> L[(Vector DB Index\nReady for search)]

    style F fill:#c8e6c9
    style G fill:#fff9c4
    style H fill:#ffccbc
```

---

## 5. Model Proposals Summary (Workbook 4.1)

| # | Model | Architecture | Key Feature | Evaluation Metrics |
|---|-------|-------------|-------------|-------------------|
| 1 | **CLIP Zero-Shot** | ViT-B/32 Image + Text Transformer | No training needed | R@1, R@5, R@10, Median Rank |
| 2 | **BLIP Zero-Shot** | Vision-Language Encoder-Decoder | Richer caption semantics | R@K + Caption BLEU |
| 3 | **Flamingo-Lite** | Cross-modal Temporal Transformer | 3-frame temporal reasoning | R@K + Temporal IoU |
| 4 | **Spartan-VLM** | CLIP backbone + LSTM | Lightweight fine-tunable hybrid | R@K + Fine-tune Loss |
| 5 | **MKD (Distilled)** | Teacher→Student Knowledge Distillation | Near-SOTA, deployable size | R@K + KL Divergence + Latency |

---

## 6. Model Comparison (Workbook 4.3)

```mermaid
quadrantChart
    title Model Comparison — Performance vs Deployment Cost
    x-axis Low Cost --> High Cost
    y-axis Low Performance --> High Performance
    quadrant-1 Ideal
    quadrant-2 Research-Only
    quadrant-3 Poor Choice
    quadrant-4 Fast Baseline
    CLIP Zero-Shot: [0.15, 0.55]
    BLIP Zero-Shot: [0.30, 0.65]
    Flamingo-Lite: [0.60, 0.75]
    Spartan-VLM: [0.40, 0.60]
    MKD Student: [0.45, 0.80]
```

---

## 7. System Support Environment (Workbook 5.4)

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11 | All pipeline and API code |
| Deep Learning | PyTorch 2.3, torchvision | Model inference and training |
| VLM Models | HuggingFace transformers, OpenAI CLIP | Pre-trained model weights |
| Video Decoding | PyAV (FFmpeg bindings) | Lossless keyframe extraction |
| Image Processing | Pillow (PIL) | Resize, crop, JPEG save |
| Data Storage | Apache Parquet, Pandas | Columnar metadata storage |
| Vector Database | Pinecone / Chroma | Embedding similarity search |
| API Framework | FastAPI 0.109 | REST search endpoint |
| Frontend | React 18 + Vite | User search interface |
| Containerisation | Docker + Docker Compose | Reproducible environments |
| CI/CD | GitHub Actions | Automated testing and lint |
| Monitoring | Prometheus + Grafana | GPU/CPU and latency metrics |
| Version Control | Git (GitHub) | Collaborative development |
