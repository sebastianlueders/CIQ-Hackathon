# Worker contracts

prepare: approved examples -> versioned dataset manifest
train: pinned base + dataset + LoRA config -> candidate adapter
evaluate: baseline + candidate + sealed cases -> per-case evidence
package: validated candidate -> checksummed artifact bundle

Workers must report real failures and runtime identities.
