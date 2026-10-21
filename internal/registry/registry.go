// Package registry is the model registry (RFC-0003, master prompt Part 2
// §4): model identity, revision, dimensions, distance metric and egress
// policy. It reuses helix-protocol's generated Model type directly rather
// than redefining the same fields here — one canonical shape, consumed,
// not duplicated (architecture/README.md's dependency rule).
package registry

import (
	"fmt"
	"sync"

	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
)

// Key encodes a model's identity exactly the way helix-kernel's job
// contract does ("model_id:model_revision" — see register_source in
// helix-kernel), so a registry key and a job's target_model are always the
// same string for the same model.
func Key(id, revision string) string {
	return id + ":" + revision
}

// Registry is a concurrency-safe, in-memory model registry. Phase 1 has no
// persistent store for it; a deployment re-registers its models at startup.
type Registry struct {
	mu     sync.RWMutex
	models map[string]protocol.Model
}

func New() *Registry {
	return &Registry{models: make(map[string]protocol.Model)}
}

// Register validates and adds a model. Returns an error rather than
// panicking or silently accepting an unusable entry: an embed-capability
// model with no declared Dims would let a dimension mismatch slip through
// Guarded's check (see providers.Guarded) by having nothing to check against.
func (r *Registry) Register(m protocol.Model) error {
	if m.Id == "" {
		return fmt.Errorf("registry: model id must not be empty")
	}
	if m.Revision == "" {
		return fmt.Errorf("registry: model %q: revision must not be empty", m.Id)
	}
	if m.Capability == protocol.Embed && (m.Dims == nil || *m.Dims <= 0) {
		return fmt.Errorf("registry: model %q: embed capability requires a positive dims", Key(m.Id, m.Revision))
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	key := Key(m.Id, m.Revision)
	if _, exists := r.models[key]; exists {
		return fmt.Errorf("registry: model %q is already registered", key)
	}
	r.models[key] = m
	return nil
}

func (r *Registry) Get(id, revision string) (protocol.Model, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.models[Key(id, revision)]
	return m, ok
}

// GetByKey looks up a model by its combined "id:revision" key, the same
// form as a job's target_model.
func (r *Registry) GetByKey(key string) (protocol.Model, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.models[key]
	return m, ok
}

func (r *Registry) List() []protocol.Model {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]protocol.Model, 0, len(r.models))
	for _, m := range r.models {
		out = append(out, m)
	}
	return out
}
