# Model training

Training is an offline process and is not part of the deployed application.
The maintained trainer consumes sealed information-set UCT rows and produces a
standalone policy artifact for explicit review and selection.

## Target and loss

Each row contains the acting player's 659-bit observation, concrete legal
actions, strategic destination groups, and the root visit counts from 128 UCT
simulations. Group visits map to their designated representative actions and
are normalized into a probability distribution.

Training uses distributional cross-entropy over representative legal actions.
Illegal actions and non-representative members of symmetric groups are masked
before log-softmax. There is no illegal-action penalty, selected-action one-hot
loss, value output, return target, recurrence, critic, or PPO objective.

Complete trajectory roots remain together when data is divided into training
and validation sets. The trainer verifies external rows and manifests before
constructing batches, preserves natural placement frequencies, and supports
deterministic CPU checkpoint/resume.

## Model and optimizer

The trainer uses the 754,601-parameter architecture described in
[neural model](neural-model.md). The active defaults are:

- AdamW with learning rate `3e-4`, betas `0.9/0.999`, epsilon `1e-8`, and
  weight decay `1e-4`.
- Batch size 256.
- Global gradient clipping at 1.0.
- 8–50 epochs, validation patience 5, and minimum improvement `1e-4`.
- Lowest validation cross-entropy for checkpoint selection.

Checkpoints and exported artifacts are written atomically. Training reports
include cross-entropy, KL divergence, entropy, top-action agreement, placement
and role splits, gradient norms, throughput, memory, and inference latency.

## Selected artifact

The selected standalone artifact is stored locally at:

```text
runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt
```

Its SHA-256 digest is:

```text
d35196cf4513589def0ffb3c4c7c268e78652a46dab8ea41001ff2c648265203
```

Release packaging verifies those exact bytes and copies only the runtime
artifact. It does not include training data, checkpoints, optimizer state, or
reports.
