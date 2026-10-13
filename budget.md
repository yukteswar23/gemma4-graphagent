# GraphAgent: Compute & Storage Budget Plan

**Competition:** Google × Kaggle Gemma 4 Developer Agent Paper Track
**Deadline:** November 12, 2026, 11:59 PM UTC
**Plan written:** October 13, 2026 (about 4 weeks left)

> All sizes and throughput figures below are estimates. Check them against real numbers during Week 1 and update this file.

---

## 1. Available Hardware

| Resource | Spec | Assessment |
|---|---|---|
| GPU | 1× NVIDIA RTX A5000, 24 GB VRAM (Ampere), CUDA 12.8 | **Main constraint** |
| System RAM | 251 GB | Plenty |
| CPU | 64 cores | Plenty: parallel Docker tests, graph building, CPU embeddings |
| Disk `/` (NVMe) | **687 GB free** | **Primary work disk** |
| Disk `/data` | 186 GB free | Backups and archived checkpoints |
| Disk `/home` | 9.5 GB free (99% full) | **Do not use.** Move all caches off `~/` |
| Disk `/mnt/storage` | 0 free | Not usable |

---

## 2. Disk Budget (on `/`)

| Item | Estimated size | Notes |
|---|---|---|
| OS packages, CUDA, PyTorch, vLLM, Unsloth, conda envs | 30 GB | |
| Gemma 4 31B QAT checkpoint, full precision (for training) | 62 GB | Download once with `local_dir=`; don't let the HF cache keep a duplicate copy |
| Gemma 4 31B W4A16 / quantized (for inference) | 20 GB | Used for vLLM rollouts and evaluation |
| LoRA adapters and checkpoints | 10 GB | About 400 MB each; keep the last 3 and move the rest to `/data` |
| SWE-bench Verified Docker images | 150 GB | Use `--cache_level env`; prune if the disk gets tight |
| SWE-Gym subset Docker images (300–500 tasks) | 150 GB | Build in batches; run `docker system prune` between batches |
| Repo clones, Tree-sitter graphs, embedding indexes | 20 GB | For the graph tools and GraphNav-Bench |
| Trajectories, SFT data, rollout logs | 15 GB | |
| **Total planned** | **~457 GB** | |
| **Headroom** | **~230 GB** | Safety margin out of 687 GB |

### One-time setup

```bash
sudo mkdir -p /work && sudo chown $USER /work
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/work/hf
export PIP_CACHE_DIR=/work/pip
export CONDA_PKGS_DIRS=/work/conda_pkgs
export TMPDIR=/work/tmp
EOF
mkdir -p /work/{hf,pip,conda_pkgs,tmp,models,data,ckpt,repos,logs}

# Docker must store its data on / (default /var/lib/docker)
docker info | grep "Docker Root Dir"
```

If this is a shared server, confirm with the admin that you can use about 450 GB on `/`.

---

## 3. GPU Memory Budget (24 GB A5000)

| Stage | VRAM use | Feasible? | Settings |
|---|---|---|---|
| Inference, 31B W4A16 | ~17 GB weights + ~6 GB KV cache | ✅ | Max context ~16k; `--gpu-memory-utilization 0.95`; try a KV-cache dtype that saves memory |
| QLoRA SFT, 31B | ~17 GB weights + LoRA + activations | ✅ | Unsloth, sequence length 4–8k, batch 1, gradient accumulation 16, gradient checkpointing |
| Full-trajectory SFT (30k+ tokens) | > 24 GB | ❌ | Use step-level examples instead (see §4) |
| GRPO on 31B | ≫ 24 GB | ❌ | Replaced with rejection-sampling fine-tuning (expert iteration); optional cloud GRPO |

The GPU can't run inference and training simultaneously. Schedule them in alternating phases: **generate rollouts → train → generate → train**.

---

## 4. Method Changes Required by the Hardware

1. **Step-level SFT built around the Evidence Buffer.** Each training example contains *issue + Evidence Buffer summary + last k tool calls → next action*. That fits in 8k tokens. It also makes the Evidence Buffer's purpose clear: it keeps the context small.
2. **Graph-RLVR becomes rejection-sampling fine-tuning (expert iteration).** Generate N rollouts per task, keep the ones that pass the tests (giving extra weight to runs that used fewer tool calls), and fine-tune on those. Repeat 2–3 rounds. The paper can still honestly call this RL from verifiable rewards.
3. **DR-LoRA** (dynamic-rank LoRA: higher rank on attention, lower on FFN) still works under QLoRA. It's one of the ablations.
4. **Optional real GRPO:** rent 1× H100 80 GB for 10–20 hours (about $25–60) in Week 4, only if expert iteration results are strong.
5. **GraphNav-Bench scope:** 150–200 annotated issues for the deadline; present 500 as the planned full release.

---

## 5. GPU Time Budget

Throughput estimates: about 300 tokens/s output across batched requests with vLLM on an A5000; each episode is about 50 tool calls / ~15k output tokens, so **about 1–2 episodes per minute**.

| Job | GPU hours |
|---|---|
| Baseline evaluation (50-task Verified subset, no fine-tuning) | 1 |
| Teacher/seed trajectory generation for SFT data | 8 |
| SFT run #1 (about 3k step-level examples, 2 epochs) | 10 |
| Expert iteration rounds 1–3 (rollouts + training) | 3 × 12 = 36 |
| Ablations (DR-LoRA vs. uniform LoRA, buffer on/off, graph tools on/off, budget policy on/off) | 25 |
| Final evaluation (full 500 Verified + competition harness) | 12 |
| Hardware profiling (VRAM, latency, throughput) | 2 |
| **Total** | **~94 GPU hours** (fits in 4 weeks if the GPU runs overnight) |

---

## 6. Week-by-Week Plan

### Week 1: Oct 13–19 · Setup & Baseline
- [ ] Do the one-time setup in §2; confirm disk permissions and Docker root
- [ ] Download the Gemma 4 31B QAT checkpoint and serve it with vLLM; measure real VRAM use and tokens/s
- [ ] Install SWE-bench; build env images for a 50-task Verified subset
- [ ] Implement the graph tools: `get_code_neighbors`, `search_similar_code`, `get_code_subgraph` (Tree-sitter + call graph + embeddings)
- [ ] Implement the Evidence Buffer and Budget Management Policy (thresholds: exploit at >75 calls, submit at >90)
- [ ] **Milestone:** baseline resolve rate on the 50-task subset, with and without the graph tools

### Week 2: Oct 20–26 · SFT Data & First Training
- [ ] Generate seed trajectories on the SWE-Gym subset; keep only runs that pass the tests
- [ ] Convert them into step-level examples (≤8k tokens) with Evidence Buffer summaries
- [ ] QLoRA SFT run #1 with DR-LoRA
- [ ] Start GraphNav-Bench annotation: schema, gold navigation paths, minimum tool budgets (target ~75 issues)
- [ ] **Milestone:** SFT model beats the baseline on the 50-task subset

### Week 3: Oct 27–Nov 2 · Expert Iteration & Ablations
- [ ] Expert iteration rounds 1–3 (sample → filter by tests + efficiency → retrain)
- [ ] Ablation runs (§5)
- [ ] Bring GraphNav-Bench to 150–200 issues; build the difficulty taxonomy
- [ ] Start the paper: Intro, Related Work, Method
- [ ] **Decision point (Nov 1):** rent an H100 for GRPO, or not

### Week 4: Nov 3–9 · Final Evaluation & Paper
- [ ] Final evaluation: competition harness + full SWE-bench Verified
- [ ] Hardware profiling on the A5000 (24 GB is the target "consumer GPU" class)
- [ ] Write Results, Ablations, GraphNav-Bench sections; make figures
- [ ] Release GraphNav-Bench (CC-BY 4.0) and the LoRA adapter

### Buffer: Nov 10–12 · Polish & Submit
- [ ] Proofread, check citations, write the AI-usage disclosure
- [ ] **Submit by Nov 12, 11:59 PM UTC** (aim for Nov 11)

---

## 7. Risks & Fallbacks

| Risk | Fallback |
|---|---|
| `/` is shared or not writable | Use `/data` (186 GB): evaluate on the Verified subset only, use cloud eval (`sb-cli` / Modal) for the full 500 |
| 16k context too small for inference | Use a quantized KV cache, more aggressive Evidence Buffer summaries, a shorter tool-output cap |
| SFT at 8k runs out of memory | Drop to 4k sequence length, lower LoRA rank on FFN layers |
| Rollouts too slow | Fewer samples per task, smaller SWE-Gym subset, run overnight |
| Expert iteration gives no gain | Present SFT + Evidence Buffer + graph tools as the main result; expert iteration as a negative/neutral ablation |
| GraphNav-Bench annotation falls behind | Auto-generate draft navigation paths from the gold patch + call graph, then hand-verify a subset |
