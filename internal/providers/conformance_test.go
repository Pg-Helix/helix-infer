package providers

import (
	"context"
	"testing"
	"time"
)

// RunConformance exercises the behaviour every EmbeddingProvider adapter
// must have, regardless of what backs it — "per adapter" conformance tests
// (Part 2 §4). newProvider builds a fresh provider for one subtest, dims is
// its model's expected dimensionality (0 if the adapter doesn't fix one).
func RunConformance(t *testing.T, newProvider func() EmbeddingProvider) {
	t.Helper()

	t.Run("rejects empty input", func(t *testing.T) {
		p := newProvider()
		_, err := p.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: nil})
		if err == nil {
			t.Fatal("expected an error for empty input, got nil")
		}
	})

	t.Run("respects an already-cancelled context", func(t *testing.T) {
		p := newProvider()
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := p.Embed(ctx, EmbedRequest{Model: "m:1", Input: []string{"hello"}})
		if err == nil {
			t.Fatal("expected an error for an already-cancelled context, got nil")
		}
	})

	t.Run("respects a deadline that has already passed", func(t *testing.T) {
		p := newProvider()
		ctx, cancel := context.WithTimeout(context.Background(), -1*time.Second)
		defer cancel()
		_, err := p.Embed(ctx, EmbedRequest{Model: "m:1", Input: []string{"hello"}})
		if err == nil {
			t.Fatal("expected an error for an expired deadline, got nil")
		}
	})

	t.Run("returns one vector per input, same length", func(t *testing.T) {
		p := newProvider()
		res, err := p.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"a", "b", "c"}})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if len(res.Vectors) != 3 {
			t.Fatalf("got %d vectors, want 3", len(res.Vectors))
		}
		for i, v := range res.Vectors {
			if len(v) != res.Dims {
				t.Fatalf("vector %d has %d dims, result reports Dims=%d", i, len(v), res.Dims)
			}
		}
	})
}
