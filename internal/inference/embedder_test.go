package inference

import (
	"context"
	"errors"
	"testing"

	protocol "github.com/Pg-Helix/helix-protocol/gen/go"

	"github.com/Pg-Helix/helix-infer/internal/providers"
	"github.com/Pg-Helix/helix-infer/internal/registry"
	"github.com/Pg-Helix/helix-infer/internal/scheduler"
	"github.com/Pg-Helix/helix-infer/internal/usage"
)

func intPtr(i int) *int { return &i }

// TestNewEmbedder_EndToEnd is the actual composed shape a caller uses:
// register a model, build an embedder for it, call it, and see usage
// recorded against the same registry key — every piece (registry,
// providers.Mock, scheduler.Guarded, usage.Tracker) wired together, not
// just unit-tested in isolation.
func TestNewEmbedder_EndToEnd(t *testing.T) {
	reg := registry.New()
	const key = "local-mock-embed:v1"
	if err := reg.Register(protocol.Model{Id: "local-mock-embed", Revision: "v1", Capability: protocol.Embed, Dims: intPtr(8)}); err != nil {
		t.Fatal(err)
	}

	tr := usage.NewTracker()
	embedder, err := NewEmbedder(reg, key, providers.NewMock(8), scheduler.Options{MaxConcurrency: 2, Usage: tr})
	if err != nil {
		t.Fatal(err)
	}

	res, err := embedder.Embed(context.Background(), providers.EmbedRequest{Model: key, Input: []string{"hello", "world"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Vectors) != 2 || res.Dims != 8 {
		t.Fatalf("got %d vectors of dim %d, want 2 vectors of dim 8", len(res.Vectors), res.Dims)
	}
	if snap := tr.Snapshot()[key]; snap.Calls != 1 || snap.Inputs != 2 {
		t.Fatalf("usage for %q = %+v, want {Calls:1 Inputs:2}", key, snap)
	}
}

func TestNewEmbedder_RejectsUnregisteredModel(t *testing.T) {
	reg := registry.New()
	_, err := NewEmbedder(reg, "nope:v1", providers.NewMock(8), scheduler.Options{})
	if err == nil {
		t.Fatal("expected an error for an unregistered model")
	}
}

func TestNewEmbedder_CatchesRealDimensionMismatch(t *testing.T) {
	reg := registry.New()
	const key = "m:v1"
	if err := reg.Register(protocol.Model{Id: "m", Revision: "v1", Capability: protocol.Embed, Dims: intPtr(16)}); err != nil {
		t.Fatal(err)
	}
	// providers.NewMock(8) actually returns 8-dim vectors; the registry says 16.
	embedder, err := NewEmbedder(reg, key, providers.NewMock(8), scheduler.Options{MaxConcurrency: 1})
	if err != nil {
		t.Fatal(err)
	}
	_, err = embedder.Embed(context.Background(), providers.EmbedRequest{Model: key, Input: []string{"x"}})
	if !errors.Is(err, providers.ErrDimMismatch) {
		t.Fatalf("got %v, want a dimension-mismatch error", err)
	}
}
