# Shape manifests

Shape rows must come from the actual streaming graph/trace, not architecture
guesses. Each row records phase, operation/tensor family, dimensions, dtype,
frequency per feed, `stream_att_right`, backend dispatch, kernel name, device
placement, and fallback state. Use `shape-manifest.schema.json` as the contract.
