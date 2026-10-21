// Package inference is the thin composition layer a caller (eventually
// C1's worker) actually uses: look a model up in the registry, wrap its
// provider with the operational guarantees in scheduler.Guarded, using the
// registry's own declared dims as the expected-dimension check rather than
// requiring every caller to repeat that lookup.
package inference

import (
	"fmt"

	"github.com/Pg-Helix/helix-infer/internal/providers"
	"github.com/Pg-Helix/helix-infer/internal/registry"
	"github.com/Pg-Helix/helix-infer/internal/scheduler"
)

// NewEmbedder builds a guarded embedding provider for modelKey ("id:revision",
// see registry.Key). It fails fast if the model is not registered, is not
// an embedding-capability model, or carries no declared dims — a Guarded
// with ExpectedDims=0 would silently skip the dimension check RFC-0003
// calls for.
func NewEmbedder(reg *registry.Registry, modelKey string, provider providers.EmbeddingProvider, opts scheduler.Options) (*scheduler.Guarded, error) {
	m, ok := reg.GetByKey(modelKey)
	if !ok {
		return nil, fmt.Errorf("inference: model %q is not registered", modelKey)
	}
	if m.Dims == nil || *m.Dims <= 0 {
		return nil, fmt.Errorf("inference: model %q has no usable dims", modelKey)
	}
	opts.ExpectedDims = *m.Dims
	return scheduler.NewGuarded(provider, opts), nil
}
