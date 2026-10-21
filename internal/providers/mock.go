package providers

import (
	"context"
	"crypto/sha256"
)

// Mock is a deterministic, local embedding provider: no network call, no
// API key, no paid dependency — what the Phase 1 local stack (RFC-0005)
// runs instead of a real model. Same input always produces the same
// vector, which is what a conformance/golden test needs; it is not a real
// embedding and carries no semantic meaning.
type Mock struct {
	Dims int
}

func NewMock(dims int) *Mock {
	return &Mock{Dims: dims}
}

func (m *Mock) Embed(ctx context.Context, req EmbedRequest) (*EmbedResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(req.Input) == 0 {
		return nil, ErrEmptyInput
	}

	vectors := make([][]float32, len(req.Input))
	for i, text := range req.Input {
		vectors[i] = deterministicVector(text, m.Dims)
	}
	return &EmbedResult{Model: req.Model, Vectors: vectors, Dims: m.Dims}, nil
}

// deterministicVector derives a vector from a SHA-256 of the text: no
// claim of semantic similarity, only of being the same output for the same
// input every time, on every machine.
func deterministicVector(text string, dims int) []float32 {
	v := make([]float32, dims)
	if dims == 0 {
		return v
	}
	h := sha256.Sum256([]byte(text))
	for i := range v {
		b := h[i%len(h)]
		v[i] = (float32(b)/255.0)*2 - 1 // normalized to [-1, 1]
	}
	return v
}
