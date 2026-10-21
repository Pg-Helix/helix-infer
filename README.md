# helix-infer

Model registry, provider routing, inference, batching, budgets and native adapters.

| | |
|---|---|
| Ownership group | AI and Retrieval |
| Primary form | Go |
| Visibility | PUBLIC |
| Status | **I1: provider interface.** Embedding provider interface, model registry, mock and OpenAI-compatible adapters exist; no service/CLI wiring yet. |

Design source of truth: the private `helix-atlas` repository (`architecture/repos/helix-infer.md`,
`rfcs/0003-pseudo-column-lifecycle-and-worker-protocol.md`).
Organization master prompt: PgHelix Organization Master Prompt, 16 Oct 2026.

## Layout

- `internal/registry/` — model registry (identity, revision, dims, metric, egress policy), reusing
  helix-protocol's generated `Model` type directly rather than redefining the same fields.
- `internal/providers/` — the versioned `EmbeddingProvider` interface, a deterministic `Mock` adapter
  (no network, no API key — what the Phase 1 local stack runs), and `OpenAICompat`, an adapter for any
  OpenAI-compatible `/embeddings` endpoint. `RunConformance` is the shared per-adapter test suite both
  implementations are run against.
- `internal/scheduler/` — `Guarded`, which wraps any `EmbeddingProvider` with a concurrency limit, a
  circuit breaker, an embedding-dimension check, and usage accounting.
- `internal/usage/` — in-memory per-model call/input counters.
- `internal/inference/` — `NewEmbedder`, the composition a caller actually uses: look a model up in the
  registry and build a `Guarded` provider for it with the registry's own dims as the expected-dimension
  check.

## Build and test

```
go build ./...
go vet ./...
go test ./... -race -cover
```

Requires Go 1.26+. No network access needed — `OpenAICompat` is tested entirely against
`httptest.Server`, never a real provider.

## License

Not yet decided. The license is an ADR in `helix-atlas`; until it is accepted, no license is granted and all rights are reserved.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
