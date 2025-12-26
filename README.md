# WTC (Who's That Character) - Technical Documentation
## Character Attribute Extraction from Anime Images

**Version**: 0.6.2 (Scaling-Ready Pipeline)  
**Last Updated**: December 26, 2025  
**Status**: Production Ready (v0.5.1 Demo) | Scaling Ready (v0.6.2 Pipeline)

---

## Executive Summary

**WTC** is a two-stage, multi-model anime character attribute extraction system designed to process large-scale image datasets (5M+ images) and extract structured character attributes for training generative models. The system uses a modular architecture combining **tag-based inference** (DeepDanbooru) and **zero-shot vision-language models** (CLIP), routing to gender-specific attribute inference modules for high accuracy.

### Key Features
- ✅ **Multi-model support**: DeepDanbooru (API), CLIP (local cached embeddings), or both for comparison
- ✅ **Gender-aware routing**: Hot-swappable functions for male/female/non-binary inference
- ✅ **Multi-modal age inference**: Clothing context + body development + facial maturity indicators
- ✅ **Streaming pipeline**: HuggingFace dataset streaming with per-shard processing
- ✅ **Colab-optimized**: Session timeout mitigation, cache clearing, JSONL streaming writes
- ✅ **Error resilience**: Per-image error logging in JSONL output, no pipeline failure
- ✅ **Production quality**: 10 of 11 attributes production-ready; skin tone & body type best-effort

### Deliverables
- **v0.5.1**: Interactive Gradio UI for single-image demo
- **v0.6.2**: Batch pipeline with HuggingFace dataset streaming, JSONL output, ready for 5M+ images
- **Ray Parallelization**: Optional distributed processing (4 GPU workers → ~17 hours for 5M images)

---

## Architecture Overview

```
INPUT (Image)
    ↓
STAGE 1: Tagging (DeepDanbooru or CLIP)
    ↓ (Raw tags: {"1girl": 0.99, "blueyes": 0.92, "long hair": 0.88, ...})
    ↓
STAGE 2: Projector (Tags → Structured Attributes)
    ├─ Character Count Gate (multi-char filter)
    ├─ Gender Detection (1girl/1boy → female/male)
    ├─ Attribute Extraction (hair, eye, body type, clothing, etc.)
    ├─ Gender-Aware Routing (female body type ≠ male body type)
    ├─ Multi-Modal Inference (age: tags + clothing + body development)
    └─ Confidence Resolution (delta threshold = 0.3 to resolve ambiguity)
    ↓
OUTPUT (Structured JSON with confidences + evidence)
    {
      "age": "young adult" (0.89),
      "gender": "female" (0.98),
      "hairColor": "black" (0.95),
      ...
    }
```

---

## Stage 0: Dataset & Metadata Architecture

### Dataset Structure
- **Source**: HuggingFace [cagliostrolab/860k-ordered-tags](https://huggingface.co/datasets/cagliostrolab/860k-ordered-tags)
- **Total**: 860,000 anime character images (~120-130 shards)
- **Shard Size**: ~10,800 MB per shard (~10,000 images)
- **Format**: PNG/JPG images with pre-split shards in dataset

### Metadata Schema
- **Index File**: `stage1_index.csv` (or `metadata_lat.json`)
- **Per-Image Fields**:
  - `image_id`: Unique identifier
  - `width`, `height`: Image resolution
  - `tags`: Pre-existing metadata tags (e.g., from Danbooru)
  - `source`: Original dataset source

### Error Handling Strategy (Stage 0)
| Error Type | Handling |
|-----------|----------|
| **Corrupted image** | Skip, log `"error": "Image decode failed"` in JSONL |
| **Missing metadata** | Use `None` for width/height, `[]` for tags |
| **Unreadable file** | Log error, continue to next image |
| **Non-image file** | Skip (pre-filtered in shard loading) |

**Implementation**: All errors logged in output JSONL with `"error"` field. Pipeline continues without failure.

---

## Stage 1: Tagging Model Selection & Evaluation

### Model Comparison Study

**Methodology**: 100-image test set sampled from 9 shards (3 start, 3 middle, 3 end)

| Model | Strength | Weakness | Use Case |
|-------|----------|----------|----------|
| **DeepDanbooru** (trained) | ✓ Excellent character count, hair, clothing detail | Limited age/body type inference | Primary tagger (default) |
| **CLIP** (zero-shot) | ✓ Strong on age, body type, expression, multi-char detection | Slower (10-15 sec/img without optimization), expensive prompts | Secondary comparator |
| **WD-14** | ✓ Fast tagging | ✗ Too many low-confidence tags, unreliable threshold | Not used in final system |
| **Metadata tags** | ✓ Ground truth, available | Limited coverage, sparse for some attributes | Validation only |

### Threshold Selection
- **Initial**: 0.5 (too strict, many low-quality images failed)
- **Final**: **0.1–0.2** (trade-off for noisy image recovery)
  - DeepDanbooru: 0.1 primary, 0.25 fallback for eye color
  - CLIP: 0.15 primary threshold

### Inference Latency
| Model | Latency | Notes |
|-------|---------|-------|
| DeepDanbooru API (Gradio Space) | 50–200 ms | Highly variable; rate limit 5–8 hits/sec |
| DeepDanbooru local (if fine-tuned) | **<50 ms** | 20 images/sec potential (not implemented) |
| CLIP (single) | 200–500 ms | Initial; optimized to ~50 ms with batch + cache |
| CLIP (batch=16) | ~50 ms avg | 320 images/batch on V100 GPU |

### Future Optimization: CLIP Fine-Tuning

**Proposed**: Train CLIP on tags-as-image-embeddings + images
- **Input**: Tag probability vector (high-dim) + image
- **Output**: Structured attributes directly
- **Benefit**: ~10× speedup, better consistency than zero-shot

**Math**:
```
Current CLIP pipeline:
  - Load model: 1 forward pass per attribute category
  - Compute image features: 1× per image
  - Compare with cached text embeddings: K comparisons (K = # categories)
  Cost per image ≈ 200-500ms (overhead from Gradio + network latency)

Fine-tuned CLIP:
  - Single forward pass: image → attributes directly
  Cost per image ≈ 50-100ms (local inference only)

Speedup ≈ 5-10×
```

---

## Stage 2: Projector (Tags → Structured Attributes)

### Architecture: Tag-to-Attribute Mapping

**Input**: `tag_probs: Dict[str, float]` (e.g., `{"1girl": 0.99, "blueyes": 0.92, ...}`)  
**Output**: `attributes: Dict[str, Any]` (structured JSON with confidences + evidence)

### 2.1 Multi-Character Gating (First Pass)

```python
estimateCharacterCount(tags) → (count: int, is_ambiguous: bool)

1. Check "solo" tag → (1, False)
2. Regex extract count from "Ngirl", "Nboy" → (N, is_multi_keyword)
3. Check multi_keywords ["multiple", "group", "couple", ...] → (2, True)
4. Default (no 1girl/1boy found) → (2, ambiguous=True)

If count ≠ 1:
  Return {"imagestatus": "ambiguousmulticharacter", "charactercount": count, "attributes": None}
  (Skip full attribute extraction)
```

**Character Count Regex**: `r'(\d+)(girl|girls|boy|boys)'` (case-insensitive)

### 2.2 Gender Detection (Router)

```python
gender = extract_attribute(tag_probs, {"1girl", "1boy"})
  → female (if "1girl")
  → male (if "1boy")
  → unknown (default)

Used to ROUTE subsequent inference:
  ├─ Female → inferFemaleBodyType() [breast size + hip size + proportions]
  ├─ Male   → inferMaleBodyType() [muscle tags + facial hair + build]
  └─ Gender-aware age inference
```

### 2.3 Gender-Aware Body Type Inference

**Female Body Type** (from breast size + hip size + proportions):

| Breast Size Tag | Weight Mapping |
|-----------------|---|
| `flatchest` | slim (0.8), petite (0.6) |
| `smallbreasts` | slim (0.5), petite (0.4) |
| `mediumbreasts` | slim (0.3), average (0.5) |
| `largebreasts` | curvy (0.6), voluptuous (0.4) |
| `hugebreasts` | voluptuous (0.8), curvy (0.5) |
| `giganticbreasts` | voluptuous (0.9) |

**Male Body Type** (from muscle + facial hair + build):
- `muscularmale` → muscular (0.9)
- `pectorals` → muscular (0.6)
- `abs` → athletic (0.7)
- `bara` → muscular (0.9), bulky (0.7)
- `dadbod` → average (0.6), chubby (0.4)
- `skinnymale` → slim (0.8), skinny (0.9)

**Confidence Resolution** (Competition Function):
```python
If multiple body type candidates (curvy, slim, muscular):
  1. Filter by HIGHCONF threshold (0.1)
  2. Sort by confidence descending
  3. If top 2 differ by < DELTA (0.3): ambiguous, pick top
  4. Else: clear winner
```

### 2.4 Multi-Modal Age Inference

**Female Age** (order of priority):
1. **Direct age tags** (2.0× weight): `teen`, `child`, `youngadult`, `middleaged`, `elderly`
2. **Breast size anime heuristic**: `flatchest` → teen (0.4), `oppailoli` → child (0.8)
3. **Clothing age hints**: `schooluniform` → teen (0.7), `businesssuit` → young adult (0.55)
4. **Facial maturity tags**: `mature` → middle-aged (0.6), `milf` → middle-aged (0.7)
5. **Height/proportion hints**: `tall` → young adult (0.3), `petite` → teen (0.3)

**Male Age** (stronger facial hair signal):
1. **Direct age tags** (2.0× weight)
2. **Facial hair** (1.5× weight): `beard` → middle-aged (0.7), `whitebeard` → elderly (0.85)
3. **Facial features**: `wrinkles` → middle-aged (0.7), `roundface` → teen (0.5)
4. **Body build**: `muscular` → young adult (0.4), `dadbod` → middle-aged (0.7)
5. **Clothing**: `gakuran` → teen (0.7), `salaryman` → middle-aged (0.75)

### 2.5 Full Attribute Mapping

**Production Quality** (reliable from tags):
- Age ✅
- Gender ✅
- Hair Color ✅
- Hair Length ✅
- Hair Style ✅
- Eye Color ✅
- Dress (category + item list) ✅
- Accessories ✅
- Anime Origin ✅

**Best-Effort** (tag inference limitations):
- Body Type ⚠️
- Skin Tone ⚠️
- Expression ⚠️

**Not Implemented**:
- **Ethnicity**: ❌ Removed (unreliable from tags; requires labeled training data + CLIP fine-tuning)
- **Scars/Tattoos**: ❌ Not extracted (requires object detection)

### 2.6 Configuration Constants

```python
DELTA = 0.3                    # Confidence difference to break ties
HIGHCONF = 0.1                 # Minimum confidence threshold
EYEFALLBACKTHRESHOLD = 0.25    # Fallback for eye color

MAXHAIRSTYLES = 2
MAXACCESSORIES = 5
MAXCLOTHINGITEMS = 5
MAXEXPRESSIONS = 3

CHARCOUNTREGEX = r'(\d+)(girl|girls|boy|boys)'
MULTIKEYWORDS = ["multiple", "several", "group", "couple", "team"]
```

---

## Single-Image Flow (v0.5.1 Demo)

### Gradio UI Features
- Image upload with preview
- Multi-tagger selection (DeepDanbooru | CLIP | Both)
- Tab-based output: Simple JSON | Full JSON | Table | Raw Tags
- Performance metrics
- Error handling

### Example Output (Simple JSON)

```json
{
  "Age": "Young Adult",
  "Gender": "Female",
  "Hair Style": "Ponytail",
  "Hair Color": "Black",
  "Hair Length": "Long",
  "Eye Color": "Blue",
  "Body Type": "Slim",
  "Dress": "Casual"
}
```

---

## Batch Pipeline (v0.6.2 Scaling)

### Overview

**v0.6.2** extends v0.5.1 with:
- ✅ HuggingFace dataset streaming
- ✅ Batch processing for CLIP (16 images)
- ✅ DeepDanbooru rate limiting (5 hits/sec)
- ✅ JSONL streaming output
- ✅ Metadata integration
- ✅ Colab optimization

### JSONL Output Schema

```json
{
  "image_id": "abc123",
  "version": "0.6.2",
  "tagger": "DeepDanbooru",
  "inference_time_sec": 0.18,
  "attributes": { /* full projector output */ },
  "metadata": {
    "width": 1280,
    "height": 720,
    "metadata_tags": ["1girl", "solo", ...]
  },
  "raw_tags": {"1girl": 0.99, "solo": 0.97, ...},
  "error": null
}
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Images/sec (DeepDanbooru)** | 5 |
| **Images/sec (CLIP)** | ~20 |
| **Memory per shard** | ~2–4 GB |
| **Processing time/shard** | 30–60 min (DD), 8–15 min (CLIP) |

---

## Scaling to 5M+ Images

### Throughput Analysis

**Single GPU**:
| Tagger | Images/Sec | Total Time (5M) |
|--------|-----------|---|
| DeepDanbooru (API) | 5 | 290 hours (12 days) |
| CLIP (batch=16) | 20 | 70 hours (3 days) |

**Ray Distributed (4 GPU Workers)**:
| Setup | Throughput | Total Time (5M) |
|-------|---|---|
| 4× DeepDanbooru | 20 img/sec | 70 hours |
| 4× CLIP | 80 img/sec | 17.5 hours |

### Ray Parallelization

```python
@ray.remote(num_gpus=1)
def process_shard_remote(shard_idx, tagger_choice, output_dir):
    shard_data = load_shard(shard_idx)
    output_jsonl = Path(output_dir) / f"shard_{shard_idx}.jsonl"
    successful, errors, _ = process_shard_batch(
        shard_data, tagger_choice, output_jsonl_path=str(output_jsonl)
    )
    return {"shard": shard_idx, "successful": successful, "errors": errors}

# Math: 4 workers × 30 shards each = 120 total
# Speedup ≈ 4× vs single GPU
```

---

## Colab-Free Constraints & Solutions

| Constraint | Solution |
|-----------|----------|
| **Session Timeout** (12 hr) | ✅ Restart + resume with checkpoint |
| **GPU Availability** (T4/K80) | ✅ Migrate to Colab Pro or local |
| **RAM Cap** (12 GB) | ✅ Streaming + cache clearing |
| **Disk Space** (100 GB) | ✅ JSONL streaming, save to Drive |
| **GPU Memory** (15 GB) | ✅ Reduce batch_size (4 instead of 16) |

### Implementation

```python
# Strategy 1: Resume-friendly pipeline
def run_pipeline_with_resume(num_shards=120, resume_from_shard=0):
    for shard_idx in range(resume_from_shard, num_shards):
        # ... process ...
        with open("checkpoint.json", 'w') as f:
            json.dump({"last_shard": shard_idx}, f)

# Strategy 2: Memory cleanup
import gc
del shard_data
gc.collect()
torch.cuda.empty_cache()

# Strategy 3: Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')
output_dir = '/content/drive/My Drive/wtc_output'
```

---

## Known Limitations & Future Work

1. **Ethnicity Inference** ❌ (requires labeled dataset + CLIP fine-tuning)
2. **Scars/Tattoos** ❌ (requires object detection)
3. **Body Type Accuracy** ⚠️ (relies on breast/hip tags)
4. **CLIP Latency** ⚠️ (200-500 ms; fine-tuning → 50 ms)

---

## Code Structure

```
WTCv0_5_1.ipynb
├─ optimizedpaste.py (Projector)
├─ cliptagger.py (CLIP)
└─ app.py (Gradio UI)

WTCv0_6_2.ipynb (adds)
├─ pipeline_utils.py (Shard loading)
├─ distributed.py (Ray)
└─ run_pipeline_v062() (Orchestrator)
```

---

## References

- **Dataset**: [cagliostrolab/860k-ordered-tags](https://huggingface.co/datasets/cagliostrolab/860k-ordered-tags)
- **DeepDanbooru**: HF Space `hysts/DeepDanbooru`
- **CLIP**: `openai/clip-vit-large-patch14`

---

**Document Version**: 1.0  
**Last Updated**: December 26, 2025, 1:35 PM IST  
