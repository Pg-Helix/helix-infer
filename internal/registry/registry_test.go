package registry

import (
	"testing"

	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
)

func intPtr(i int) *int { return &i }

func TestRegistry_RegisterAndGet(t *testing.T) {
	r := New()
	m := protocol.Model{Id: "mock-embed", Revision: "v1", Capability: protocol.Embed, Dims: intPtr(8)}
	if err := r.Register(m); err != nil {
		t.Fatal(err)
	}
	got, ok := r.Get("mock-embed", "v1")
	if !ok {
		t.Fatal("expected to find the registered model")
	}
	if got.Id != m.Id || *got.Dims != 8 {
		t.Fatalf("got %+v, want %+v", got, m)
	}
	if got, ok := r.GetByKey(Key("mock-embed", "v1")); !ok || got.Id != "mock-embed" {
		t.Fatalf("GetByKey returned %+v, %v", got, ok)
	}
}

func TestRegistry_RejectsDuplicates(t *testing.T) {
	r := New()
	m := protocol.Model{Id: "m", Revision: "v1", Capability: protocol.Embed, Dims: intPtr(8)}
	if err := r.Register(m); err != nil {
		t.Fatal(err)
	}
	if err := r.Register(m); err == nil {
		t.Fatal("expected an error re-registering the same id+revision")
	}
}

func TestRegistry_RejectsEmbedWithoutDims(t *testing.T) {
	r := New()
	err := r.Register(protocol.Model{Id: "m", Revision: "v1", Capability: protocol.Embed})
	if err == nil {
		t.Fatal("expected an error registering an embed-capability model with no dims")
	}
}

func TestRegistry_RejectsMissingIdentity(t *testing.T) {
	r := New()
	if err := r.Register(protocol.Model{Revision: "v1"}); err == nil {
		t.Fatal("expected an error for a missing id")
	}
	if err := r.Register(protocol.Model{Id: "m"}); err == nil {
		t.Fatal("expected an error for a missing revision")
	}
}

func TestRegistry_List(t *testing.T) {
	r := New()
	_ = r.Register(protocol.Model{Id: "a", Revision: "v1", Capability: protocol.Embed, Dims: intPtr(4)})
	_ = r.Register(protocol.Model{Id: "b", Revision: "v1", Capability: protocol.Generate})
	if got := len(r.List()); got != 2 {
		t.Fatalf("List() returned %d models, want 2", got)
	}
}
