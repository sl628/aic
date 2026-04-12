# aic_rlt — RL Token (RLT) Implementation

Implementation of **"RL Token: Bootstrapping Online RL with Vision-Language-Action Models"**
(Xu et al., Physical Intelligence, 2025). [[Paper]](https://www.pi.website/download/rlt.pdf)

---

## Usage

### 1. Install

```bash
cd aic_utils/aic_rlt
pip install -e .
```

### 2. Implement the VLA stub

In `scripts/train.py`, replace `load_vla()` with your actual VLA (π0, ACT, etc.).
The VLA object must expose:

```python
vla.get_embeddings(obs) -> torch.Tensor  # (1, N, D_vla) internal embeddings
vla.get_action_chunk(obs) -> np.ndarray  # (C, D) reference action chunk
```

For ACT (the AIC baseline), add forward hooks on the transformer layers to capture
intermediate embeddings and update `RLTokenConfig.vla_embed_dim` / `num_vla_tokens`
to match.

### 3. Pre-extract XVLA embeddings

Before Phase 1 training you must run `prepare_embeddings.py` once to extract XVLA
Florence-2 embeddings from your LeRobot v3.0 dataset and save them to disk.

```bash
pixi run python scripts/prepare_embeddings.py \
    --data_dir /home/yifeng/aic_data \
    --model_dir /home/yifeng/models/xvla-base \
    --output_dir /home/yifeng/aic_data/embeddings \
    --camera center_camera \
    --batch_size 8
```

This reads every frame from the dataset's parquet files, runs each image through the
XVLA encoder, and writes one file per episode:

```
<output_dir>/episode_0000.pt  →  {"vla_embeddings": Tensor(T, num_tokens, 1024)}
<output_dir>/episode_0001.pt  →  ...
```

The script is idempotent — already-extracted episodes are skipped unless `--overwrite`
is passed.

### 4. Phase 1 — Pretrain RL Token

Requires: the embeddings directory produced in step 3.

```bash
python scripts/train.py \
    --mode pretrain_rl_token \
    --data_dir /home/yifeng/aic_data \
    --embeddings_dir /home/yifeng/aic_data/embeddings \
    --checkpoint_dir checkpoints/rlt
```

Embedding dimensions (`vla_embed_dim`, `num_vla_tokens`) are detected automatically
from the first `.pt` file in `--embeddings_dir`.

### 5a. Phase 2 — Offline RL (trajectory data only)

Requires: the Phase 1 checkpoint and the embeddings directory. Increase `--bc_coeff`
to 2.0–5.0 to compensate for the lack of online coverage.

```bash
python scripts/train.py \
    --mode offline_rl \
    --load_checkpoint checkpoints/rlt/phase1_rl_token.pt \
    --data_dir /home/yifeng/aic_data \
    --embeddings_dir /home/yifeng/aic_data/embeddings \
    --checkpoint_dir checkpoints/rlt \
    --bc_coeff 3.0 \
    --n_offline_epochs 100
```

### 5b. Phase 2 — Online RL (live robot or simulator)

Requires: real-time environment interaction. Replace `AICEnvWrapper` in `train.py` with
real environment calls (MuJoCo, Gazebo, or the live robot via the AIC ROS 2 interface).

```bash
python scripts/train.py \
    --mode online_rl \
    --load_checkpoint checkpoints/rlt/phase1_rl_token.pt \
    --vla_model_dir /home/yifeng/models/xvla-base \
    --checkpoint_dir checkpoints/rlt \
    --n_warmup_steps 2000 \
    --total_env_steps 50000 \
    --bc_coeff 1.0
```

### 5. Inference (AIC ROS 2 framework)

```bash
pixi run ros2 run aic_model aic_model \
    --ros-args \
    -p use_sim_time:=true \
    -p policy:=aic_example_policies.ros.RunRLT \
    -p policy_args.checkpoint_path:=/path/to/checkpoints/rlt/final.pt
```

Fill in `RunRLT._load_vla()` with the same VLA loader used during training.

---

## Overview

RLT is a lightweight method for fine-tuning a pretrained Vision-Language-Action (VLA) model
with online reinforcement learning. It improves the hardest phases of manipulation tasks
(e.g. sub-millimeter cable insertion) in a few hours of real-robot practice.

The key idea is to avoid training the full VLA with RL (expensive and unstable). Instead:

1. **RL Token** — train a small encoder-decoder on top of the frozen VLA to compress its
   internal embeddings into a single compact vector `z_rl` (the "RL token").
2. **Actor-Critic** — train lightweight MLP actor and critic networks that take `z_rl` +
   proprioception as state, conditioned on the VLA's own action prediction as a reference.

---

## Architecture

```
Observation
    │
    ▼
Frozen VLA backbone (e.g. π0: SigLIP 400M + Gemma 4B)
    │  internal embeddings (N × D_vla)        reference action chunk ā (C × D)
    │                                                        │
    ▼                                                        │
RL Token Encoder (Transformer, frozen after Phase 1)        │
    │  z_rl  (1 × 2048)                                     │
    │                                                        │
    └────────────┬───────────────────────────────────────────┘
                 │  RL state x = (z_rl, s^p, ā)
                 ▼
         Actor MLP  ──►  action chunk a (C × D)
         Critic MLP ──►  Q(x, a)
```

**Dimensions (defaults):**

| Symbol | Meaning | Default |
|--------|---------|---------|
| `D_vla` | VLA embedding dim | 7848 |
| `N` | VLA tokens per obs | 540 |
| `D_rl` | RL token dim (`z_rl`) | 2048 |
| `C` | Action chunk length | 10 |
| `D` | Action dim (6D vel + gripper) | 7 |
| `prop_dim` | Proprioceptive state dim | 26 |

---

## How the RL Token Is Generated

### The Problem

A VLA like π0 has billions of parameters spread across many transformer layers. Each layer
produces a sequence of embedding vectors — one per input token (image patches + language
tokens) — resulting in a matrix of shape `(N, D_vla)`, e.g. `(540, 7848)`. This is far
too large to feed directly into a lightweight RL actor/critic.

The RL token is a **bottleneck compression** of all those embeddings into a single
2048-dim vector `z_rl`.

---

### Step-by-Step: Encoder

**1. VLA produces internal embeddings (stop-gradient)**

The frozen VLA processes an observation and exposes its internal transformer layer outputs.
These are immediately detached — the VLA is never modified:

```python
x = self.input_proj(vla_embeddings.detach())  # (B, N, D_enc)
```

A linear projection shrinks each token from `D_vla=7848` → `D_enc=512`, the encoder's
working dimension.

**2. A learnable readout token is appended**

```python
readout = self.readout_embed.expand(B, -1, -1)  # (B, 1, D_enc)
x = torch.cat([x, readout], dim=1)              # (B, N+1, D_enc)
```

`readout_embed` is a single learned vector (trained with the encoder). It acts as a
"query slot" that will attend to all N VLA tokens and summarize them into one vector.

**3. Lightweight encoder transformer runs**

```python
x = self.pos_enc(x)   # sinusoidal positional encoding
x = self.encoder(x)   # 4-layer Transformer (self-attention), output: (B, N+1, D_enc)
```

Self-attention lets every token — including the readout — attend to every other token.
After 4 layers, the readout position has aggregated information from the entire VLA
embedding sequence.

**4. The readout position IS the RL token**

```python
readout_out = x[:, -1, :]             # (B, D_enc) — last position = readout
z_rl = self.readout_proj(readout_out) # (B, D_rl=2048)
```

Only the output at the readout token position is kept. A final linear projection maps
it to `D_rl=2048`. This is `z_rl` — the RL token.

---

### Why Does It Capture Useful Information? The Decoder

A readout token trained in isolation could collapse to a trivial constant. The decoder
prevents this by forcing `z_rl` to be **sufficient to reconstruct the original VLA
embeddings** (the bottleneck principle):

```python
memory = self.z_rl_to_memory(z_rl).unsqueeze(1)        # (B, 1, D_enc)
decoded = self.decoder(queries, memory, tgt_mask=causal) # (B, N, D_enc)
reconstructed = self.output_proj(decoded)                # (B, N, D_vla)
```

The decoder is a **cross-attention transformer**: it generates N output vectors by
attending to `z_rl` (the sole memory) at each of N positions. The causal mask makes
reconstruction **autoregressive** — predicting token `i` from tokens `0..i-1`, like a
language model. The training loss is MSE (equation 2 in the paper):

```
L_ro = E[ Σ_i || decoder(z_rl)_i − vla_embedding_i ||² ]
```

Because the decoder must reconstruct *all* N tokens from only `z_rl`, the encoder is
forced to pack everything task-relevant into that single vector.

After Phase 1 training, **the decoder is discarded**. Only the encoder is kept (frozen)
for Phase 2, where `z_rl` is extracted at every timestep and fed into the actor/critic.

---

### Cross-Attention in the Decoder

In a standard **self-attention** layer, a sequence attends to **itself** — every token
queries every other token in the same sequence. **Cross-attention** is different: there
are two separate sequences with distinct roles.

| Role | Name | What it is here |
|------|------|-----------------|
| **Query** | What we want to produce | N positional query vectors (one per token to reconstruct) |
| **Key / Value** | The information source | `z_rl` — the RL token (just 1 vector) |

Each query says: *"given my position, what should I pull out of `z_rl`?"* The attention
mechanism computes a weighted read from the Key/Value for each query position independently:

```
Attention(Q_i, K, V) = softmax(Q_i · K^T / √d) · V
```

where `K` and `V` are both derived from the single `z_rl` vector.

Self-attention vs. cross-attention side by side:

```
SELF-ATTENTION (encoder):
  Q, K, V all come from the same sequence
  [tok_0, tok_1, ..., tok_539, readout]  ← attends to itself

CROSS-ATTENTION (decoder):
  Q   comes from: positional query vectors  (what position am I?)
  K,V come from:  z_rl                      (what information exists?)

  query_0   → reads z_rl → reconstructs tok_0
  query_1   → reads z_rl → reconstructs tok_1
  ...
  query_539 → reads z_rl → reconstructs tok_539
```

---

### How Decoder Queries Are Generated

A natural question is: what are the `queries` in the cross-attention, and are they random?

```python
queries = torch.zeros(B, num_tokens, D_enc, device=z_rl.device)
queries = self.pos_enc.pe[:, :num_tokens].expand(B, -1, -1) + queries
```

The queries are **purely positional encoding vectors** — no randomness, no learned
parameters. Since `queries` starts as all zeros, each query is simply:

```
query_i = sinusoidal_position_encoding(i)
        = [ sin(i / 10000^(0/D)),  cos(i / 10000^(0/D)),
            sin(i / 10000^(2/D)),  cos(i / 10000^(2/D)), ... ]
```

Each query carries **zero content** — only the information "I am position i". All actual
content must flow through the cross-attention read from `z_rl`. This is intentional: the
decoder has no access to the original VLA embeddings, so it cannot cheat — `z_rl` is the
only source of information to reconstruct from.

Because `z_rl` is 2048-dimensional and cross-attention uses multiple heads, each head
reads a different subspace of `z_rl`. The N query positions can therefore "ask different
questions" of the same vector to reconstruct different tokens. If `z_rl` doesn't contain
enough information, reconstruction fails — which is the training pressure that forces the
encoder to produce a rich, information-dense token.

**Note — approximation vs. strict autoregression:** The paper describes the decoder as
autoregressive (token `i` predicted from previously reconstructed tokens `0..i-1`). The
current implementation approximates this with a causal mask on fixed positional queries,
which is parallelizable but less strictly autoregressive. A fully faithful version would
feed reconstructed outputs back as inputs for the next step:

```python
# Strict autoregressive decoding (not currently implemented):
reconstructed = []
query = start_token
for i in range(num_tokens):
    out = cross_attend(query=query, key_value=z_rl)
    reconstructed.append(out)
    query = out  # feed output back as next query
```

For the purpose of training a useful bottleneck, the causal-mask approximation is
sufficient — the encoder is still forced to pack all task-relevant information into `z_rl`.

---

### Visual Summary

```
VLA embeddings (N=540, D=7848)  — frozen, stop-gradient
        │
        │  input_proj  (7848 → 512)
        ▼
[ tok_0 | tok_1 | ... | tok_539 | readout ]   (N+1, 512)
        │
        │  sinusoidal positional encoding
        │  4-layer Transformer encoder (self-attention)
        ▼
[ out_0 | out_1 | ... | out_539 | OUT_READOUT ]
                                       │
                                       │  linear  (512 → 2048)
                                       ▼
                                    z_rl          ← THE RL TOKEN

          ┌──────────────────────────────────────────┐
          │  DECODER  (Phase 1 training only)        │
          │  z_rl → cross-attention → N vectors      │
          │  output_proj → (N, 7848)                 │
          │  loss: MSE vs original VLA embeddings    │
          └──────────────────────────────────────────┘
                    (discarded after Phase 1)
```

---

## Actor and Critic MLPs

### Actor

The actor implements a Gaussian policy `π_θ(a_{1:C} | x, ā_{1:C})` that outputs an action
chunk — a sequence of `C=10` actions at once.

**Input** — three tensors concatenated into one flat vector:

```
[ z_rl (2048) | prop (26) | ref_action_chunk (C×D = 70) ]
  └─ RL token   └─ joint pos, TCP pose/vel/err   └─ VLA's suggested actions
```

**Trunk MLP** — maps the concatenated input to `C×D = 70` output values (the mean `μ`):

```
Linear → LayerNorm → SiLU  →  ...repeat...  →  Linear
                                                    │
                                             reshape to (B, C, D) = μ
```

Default: 2 hidden layers of 256 units. For harder tasks (e.g. screw insertion), use
`[512, 512, 512]`.

**Log std** — a single learned parameter vector of size `C×D`, shared across all inputs
(state-independent). It is not predicted from the observation:

```python
self.log_std = nn.Parameter(torch.zeros(C * D))   # initialized to 0 → std = 1.0
log_std = self.log_std.clamp(-5.0, 2.0)           # std range: [0.007, 7.4]
```

Initialized to 0 (std = 1.0) for broad early exploration. Learned via gradient descent
alongside the rest of the actor — shrinks as the policy becomes more confident.

**Reference action dropout** — at training time, 50% of samples have their VLA reference
chunk zeroed out:

```python
mask = torch.bernoulli(torch.full((B, 1), 0.5, ...))
ref_flat = ref_flat * mask
```

This prevents the actor from blindly copying the VLA. Without it, the actor learns to
rely on the reference signal and fails when it is unavailable (e.g. at test-time if the
VLA is unavailable, or at the next state during critic target computation).

**Sampling vs. deterministic output:**
- `sample()` — used during training; draws `action ~ N(μ, σ²)` via `rsample()`
  (reparameterization trick so gradients flow back through the sample)
- `get_mean()` — used at inference; returns `μ` directly, no stochasticity

**Note:** `sample()` also returns `log_prob`, but it is always discarded (`_`). This is
not a policy gradient algorithm — gradients flow through the reparameterized sample
into the Q-loss and BC-loss directly.

---

### Critic (Twin Q-network)

The critic estimates `Q(x, a_{1:C})` — the expected return for taking a chunk of actions
from a given state. It uses a **TD3-style twin ensemble** of two independent Q-networks
to reduce overestimation bias.

**Input** — same state, plus the action chunk:

```
[ z_rl (2048) | prop (26) | action_chunk (C×D = 70) ]  →  scalar Q-value
```

Each Q-network has the same MLP architecture as the actor trunk. The two networks have
independent weights and are trained separately.

**`min_q()`** returns `min(Q1, Q2)` — the pessimistic estimate. This is used in:
- **Actor updates**: to avoid exploiting overestimated Q-values
- **Critic target computation**: to prevent value overestimation from propagating

**Target critic** — a frozen copy of the critic updated via EMA (τ = 0.005) after every
critic gradient step. Used only for computing TD targets; never directly trained.

---

## Actor-Critic: Key Values Reference

### Networks (persistent, have weights)

| Name | What it is | How weights update |
|---|---|---|
| `actor` | The policy. Takes `(z_rl, prop, ref_action)` → outputs action chunk | Adam optimizer, via actor loss |
| `critic` | Twin Q-networks (Q1, Q2). Takes `(z_rl, prop, action)` → scalar Q-value | Adam optimizer, via critic loss |
| `critic_target` | Frozen lagged copy of `critic`. Same architecture, different weights | EMA only: `θ' ← (1-τ)θ' + τθ` |

### Values computed per update step (not stored, recomputed each batch)

| Name | What it is | Where it comes from |
|---|---|---|
| `q_target` | The TD learning label — what Q "should" equal | `reward + γ^C * critic_target.min_q(next_state, next_action)` |
| `Q1, Q2` | Current Q estimates for the batch | `critic.forward(state, action)` |
| `q_val` | Pessimistic Q used in actor loss | `critic.min_q(state, actor_action)` |

### Relationships

```
actor ──sample──► action_chunk ──► env ──► reward, next_obs
  │                                               │
  │                                         replay buffer
  │                                               │
  │◄──────────────── actor loss ◄─── q_val (min of Q1,Q2 on actor's action)
  │                                    + BC loss vs ref_action
  │
critic (Q1, Q2) ──► critic loss = MSE(Q1, q_target) + MSE(Q2, q_target)
                                              ▲
                              q_target (computed with no_grad)
                                = reward + γ^C * critic_target.min_q(next_state, ...)
                                                         ▲
                                                  critic_target
                                               (EMA copy of critic,
                                                never gradient-trained)
```

### One-line rules to keep them straight

- **`critic`** — the one being trained
- **`critic_target`** — the one used for stable labels, trails behind `critic`
- **`q_target`** — a number (tensor of scalars), not a network; the regression target for the critic loss
- **`q_val`** — a number used to push the actor toward higher-value actions

---

## Training Process

### Phase 1: RL Token Pretraining

Before any RL, the RL token encoder-decoder is trained on demonstration data:

1. Feed demo VLA embeddings through the encoder → `z_rl`
2. Decoder reconstructs the original embeddings from `z_rl` (MSE loss, equation 2)
3. Once done, **freeze** the RL token model — it is never updated again

### Phase 2: Online RL

The main loop alternates between rollout and gradient updates at a 1:5 ratio:

```
while env_step_count < total_env_steps:
    collect_transition(...)        # 1 environment step
    for _ in range(5):             # updates_per_step = 5
        gradient_update_step()     # 2 critic updates + 1 actor update
```

#### Rollout — `_collect_transition`

Three possible sources for the action chunk, in priority order:

| Condition | Action source |
|---|---|
| Human intervenes | Human-provided chunk |
| `step < 2000` (warmup) | VLA reference directly |
| Otherwise | `actor.sample(...)` |

After executing the chunk, a `Transition` `(z_rl, prop, action_chunk, ref_chunk, reward,
next_z_rl, next_prop, done)` is stored in the replay buffer.

#### Gradient Update — `_gradient_update_step`

Each update round performs **2 critic updates then 1 actor update**:

**Critic update** (equation 3) — minimize Bellman error:

```python
# TD3 target: actor mean + clipped noise, evaluated by frozen target critic
next_action = actor(next_state, ref=zeros) + clipped_noise
q_target = reward + γ^C * (1 - done) * critic_target.min_q(next_state, next_action)

# Minimize MSE for both Q-networks
critic_loss = MSE(Q1, q_target) + MSE(Q2, q_target)
```

After each critic update, the **target critic** is soft-updated: `θ' ← (1-τ)θ' + τθ`.

**Actor update** (equation 5) — maximize Q while staying close to the VLA:

```python
action = actor.sample(state, ref_action)         # reparameterized sample
actor_loss = -critic.min_q(state, action).mean() # maximize Q
           + β * MSE(action, ref_action)          # BC regularizer toward VLA
```

#### Data flow

```
obs
 ├─ VLA embeddings ──► RLTokenEncoder (frozen) ──► z_rl ─┐
 └─ proprioception ──────────────────────────────► prop  ─┴─► state x
                                                              │
                                                    ┌─────────┤
                                                    ▼         ▼
                                                  Actor     Critic
                                                    │         │
                                             action chunk   Q-value
                                                    │
                                               env_step
                                                    │
                                            reward, next_obs
                                                    │
                                            replay_buffer
                                                    │
                                        ┌───────────┴──────────┐
                                   critic update          actor update
                                  (TD Bellman loss)   (max Q + BC loss)
```

#### Key design choices

- **`γ^C` discount**: the critic targets a full chunk of `C=10` steps, so the bootstrap
  is discounted by `C` steps rather than 1.
- **No ref action at next state**: during critic target computation `ref=zeros` is passed,
  forcing the actor to handle the no-reference case — which is why ref action dropout
  during rollout is essential training for this scenario.
- **log_prob discarded**: despite using `sample()`, this is not a policy gradient method.
  Gradients flow back through `rsample()` (reparameterization trick) directly into the
  actor weights via the Q-loss. log_std is therefore trained only through Q-loss and
  BC-loss, not through an explicit entropy objective.
- **2 critic updates per 1 actor update**: standard TD3 practice — keeps the critic more
  accurate than the actor so Q-values are reliable before the actor chases them.

---

## File Structure

```
aic_utils/aic_rlt/
├── aic_rlt/
│   ├── __init__.py
│   ├── models/
│   │   ├── rl_token.py       # RL Token encoder-decoder (Section III-A)
│   │   └── actor_critic.py   # Actor + twin-critic MLPs (Section III-B)
│   ├── data/
│   │   └── lerobot_dataset.py  # LeRobotEmbeddingDataset — reads pre-extracted .pt files
│   ├── vla/
│   │   └── xvla_wrapper.py   # XVLAWrapper — XVLA model interface
│   ├── replay_buffer.py      # Off-policy replay buffer
│   └── trainer.py            # Full training loop — Algorithm 1
├── scripts/
│   ├── prepare_embeddings.py # Step 3: offline XVLA embedding extraction (run once)
│   └── train.py              # Steps 4-5: Phase 1 + Phase 2 training entry-point
├── setup.py
└── README.md                 # This file

# Inference policy (AIC ROS 2 framework):
aic_example_policies/aic_example_policies/ros/RunRLT.py
```

---

## Training — Algorithm 1

### Phase 1: RL Token Pretraining

Train the encoder-decoder on a small demonstration dataset to produce `z_rl`.
The decoder must autoregressively reconstruct the original VLA embeddings from `z_rl`,
forcing the token to capture all task-relevant information (equation 2):

```
L_ro = E[ Σ_i || h_φ(d_φ(z_{i-1}, z_{1:i-1}))_i − z_i ||² ]
```

### Phase 2: Online RL

Freeze the VLA and RL token encoder. Train actor and critic online:

**Critic** (TD3, equation 3):
```
Q̂ = Σ γ^t' r_t' + γ^C * E_{a'~π}[Q'(x', a')]
L_Q = E[(Q̂ − Q_ψ(x, a))²]
```

**Actor** (Q-maximization + BC regularizer toward VLA reference, equation 5):
```
L_a = E[ −Q_ψ(x, a) + β * ||a − ā||² ]
```

**Reference action dropout** (Section III-B): zero out the VLA reference chunk for 50%
of training samples so the actor does not become over-reliant on the reference signal.

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Chunk length C | 10 | Steps per action chunk |
| Control frequency | 50 Hz | Delta action space |
| Subsample stride | 2 | 5 transitions stored per chunk |
| Update-to-data ratio | 5 | Gradient steps per env step |
| Critic updates per actor update | 2 | TD3 convention |
| BC coefficient β | 1.0 | Tune per task |
| Target network EMA τ | 0.005 | Slow target update |
| Warmup steps N_warm | 2000 | VLA rollouts before RL |
| Hidden dims (actor/critic) | [256, 256] | [512, 512, 512] for harder tasks |
| Ref action dropout | 50% | Section III-B |

---

## Training with Offline Trajectory Data

The original RLT paper trains the actor-critic online against a live robot. If you only
have pre-collected trajectory data (no real-time robot access), the three phases below
cover what is and isn't achievable.

### What offline trajectories provide

| Available | Needed for |
|---|---|
| `(obs, action_chunk, next_obs)` tuples | Phase 1 (RL token), actor BC training |
| VLA embeddings per timestep | Phase 1 |
| Reward labels (success/failure) | Phase 2 offline RL (optional but recommended) |
| Real-time robot interaction | Phase 2 online RL (not available offline) |

### Option A — Phase 1 only (no rewards needed)

Train the RL token encoder, then fine-tune the actor with BC loss only. This improves
the actor's ability to track the VLA reference but cannot exceed the VLA's own capability.
No critic or reward signal is needed.

```python
# Actor loss: BC only, drop Q-maximization
actor_loss = β * F.mse_loss(action, ref_action)
```

Use this when you have no reward labels and want a stable fine-tuned policy.

### Option B — Offline RL (rewards required)

Pre-populate the replay buffer with your trajectory data and run the standard TD3 updates
without any online rollouts. The architecture supports this directly — omit
`_collect_transition` and load transitions from disk instead.

The main risk is **Q overestimation**: the critic will assign high values to
out-of-distribution actions it has never seen. The BC regularizer (`β * MSE`) acts as a
conservative constraint that keeps the actor close to the data distribution. Increase `β`
(e.g. 2.0–5.0) relative to the online setting to compensate.

```python
# Pre-populate replay buffer from trajectory files
for traj in offline_trajectories:
    for t in range(len(traj)):
        replay_buffer.add(Transition(
            z_rl=traj[t].z_rl,
            prop=traj[t].prop,
            action_chunk=traj[t].action_chunk,
            ref_action_chunk=traj[t].ref_action_chunk,
            reward=traj[t].reward,          # binary success or dense reward
            next_z_rl=traj[t+1].z_rl,
            next_prop=traj[t+1].prop,
            done=traj[t].done,
        ))

# Then run gradient updates directly (no rollout loop)
for _ in range(total_gradient_steps):
    trainer._gradient_update_step()
```

### Option C — Offline pre-training → online fine-tuning

Use trajectory data for Phase 1 and offline actor/critic initialization (Option B), then
continue with online RL when robot access becomes available. The offline-trained critic
provides a better starting point, reducing the online interaction needed to converge.

### Comparison

| Mode | Rewards needed | Robot needed | Exceeds VLA baseline |
|---|---|---|---|
| Phase 1 + BC only | No | No | Marginally |
| Offline RL | Yes | No | Partially (limited by data coverage) |
| Offline pre-train → online fine-tune | Yes | Eventually | Yes |
| Full online RL (paper) | Yes | Yes | Yes (best) |

---

## Programmatic API

```python
import torch
from aic_rlt import RLTConfig, RLTTrainer
from aic_rlt.models.rl_token import RLTokenConfig
from aic_rlt.models.actor_critic import ActorCriticConfig

config = RLTConfig(
    rl_token=RLTokenConfig(vla_embed_dim=2048, num_vla_tokens=256),
    actor_critic=ActorCriticConfig(hidden_dims=[256, 256], bc_coeff=1.0),
    n_warmup_steps=2000,
    total_env_steps=50000,
)

trainer = RLTTrainer(
    config=config,
    device=torch.device("cuda"),
    get_vla_embeddings=lambda obs: vla.get_embeddings(obs),
    get_vla_action_chunk=lambda obs: vla.get_action_chunk(obs),
    get_prop_state=lambda obs: env.get_prop_state(obs),
    env_step=lambda a: env.step(a),
)

trainer.pretrain_rl_token(demo_dataset)   # Phase 1
trainer.train(initial_obs_fn=env.reset)   # Phase 2
```

---

## Reference

```
@article{xu2025rlt,
  title   = {RL Token: Bootstrapping Online RL with Vision-Language-Action Models},
  author  = {Xu, Charles and Springenberg, Jost Tobias and Amin, Ali and Esmail, Adnan
             and Levine, Sergey and Ke, Liyiming},
  journal = {arXiv},
  year    = {2025},
  url     = {https://pi.website/research/rlt}
}
```
