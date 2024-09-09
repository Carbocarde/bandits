# bandits: Biased Thompson Sampling for Multi Armed Bandit

# Actions

```
new
run {config}
rank {config}
reset {config} -s {script}
summarize {config}
lint {config}
```

# Limit

This will only collect up to the limit of interesting cases before deactivating that bandit.

# Weight

Useful when certian bandits are more valuable than others.

Higher weights are prioritized more. A 10x weight is considered to run 10x faster than a 1x weight command.

```
bandits config.json
```

# Run benchmarks:

```
cargo +nightly bench --release
```

## boost::ibeta

Boost's implementation benches roughly 4x faster than `puruspe` on an m1 mac.
If this proves to be a bottleneck it might be worth using c bindings rather than using a rust library.
