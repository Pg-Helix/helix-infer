package embed

import (
	"context"
	"testing"

	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
)

// TestPublicSurfaceEndToEnd guards the re-export itself: a consumer module
// (helix-cortex) can only reach this package, not internal/ directly, so a
// breakage here would otherwise only surface downstream.
func TestPublicSurfaceEndToEnd(t *testing.T) {
	dims := 8
	reg := NewRegistry()
	if err := reg.Register(protocol.Model{Id: "mock", Revision: "v1", Capability: protocol.Embed, Dims: &dims}); err != nil {
		t.Fatalf("register: %v", err)
	}

	embedder, err := NewEmbedder(reg, RegistryKey("mock", "v1"), NewMock(dims), Options{MaxConcurrency: 1})
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}

	res, err := embedder.Embed(context.Background(), EmbedRequest{Model: "mock:v1", Input: []string{"hello"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if res.Dims != dims || len(res.Vectors) != 1 {
		t.Fatalf("unexpected result: %+v", res)
	}
}
