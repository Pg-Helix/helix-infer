// Package providers defines the versioned embedding-provider interface
// (RFC's Part 2 §4: text generation/reranking/prediction follow later —
// embeddings first) and the adapters that implement it.
package providers

import (
	"context"
	"errors"
)

// EmbedRequest is one embedding call. Model is a registry key
// ("id:revision", see registry.Key) — adapters do not interpret it beyond
// passing it through to the underlying provider/endpoint.
type EmbedRequest struct {
	Model string
	Input []string
}

// EmbedResult carries the provider's own reported dimensionality alongside
// the vectors: a mismatch against the registry's declared dims is exactly
// what Guarded checks for (an "embedding-dimension check", Part 2 §4).
type EmbedResult struct {
	Model   string
	Vectors [][]float32
	Dims    int
}

// EmbeddingProvider is the versioned interface every adapter (mock, an
// OpenAI-compatible HTTP endpoint, a future local/ONNX adapter) implements.
// Implementations must respect ctx cancellation/deadline themselves — Embed
// is expected to return promptly once ctx is done, not after finishing
// unrelated work.
type EmbeddingProvider interface {
	Embed(ctx context.Context, req EmbedRequest) (*EmbedResult, error)
}

var (
	// ErrEmptyInput is returned when a request has no input texts to embed.
	ErrEmptyInput = errors.New("providers: input must not be empty")
	// ErrDimMismatch is returned by Guarded when a provider's reported
	// dimensionality does not match the registry's declared dims for the
	// requested model.
	ErrDimMismatch = errors.New("providers: returned embedding dimensions do not match the registered model")
	// ErrCircuitOpen is returned by Guarded while its circuit breaker is open.
	ErrCircuitOpen = errors.New("providers: circuit breaker open, not calling the provider")
)
