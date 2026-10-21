// Package embed is helix-infer's public consumer surface: everything under
// internal/ is, by Go's own rule, only importable from within this module,
// so a separate module (helix-cortex's worker) cannot reach it directly.
// This package re-exports exactly what a worker needs via type aliases and
// thin constructors -- no behaviour of its own, no duplicated logic.
package embed

import (
	"github.com/Pg-Helix/helix-infer/internal/inference"
	"github.com/Pg-Helix/helix-infer/internal/providers"
	"github.com/Pg-Helix/helix-infer/internal/registry"
	"github.com/Pg-Helix/helix-infer/internal/scheduler"
)

type (
	EmbedRequest      = providers.EmbedRequest
	EmbedResult       = providers.EmbedResult
	EmbeddingProvider = providers.EmbeddingProvider
	Registry          = registry.Registry
	Options           = scheduler.Options
	Guarded           = scheduler.Guarded
)

var (
	ErrEmptyInput  = providers.ErrEmptyInput
	ErrDimMismatch = providers.ErrDimMismatch
	ErrCircuitOpen = providers.ErrCircuitOpen
)

// NewRegistry returns an empty model registry.
func NewRegistry() *Registry { return registry.New() }

// RegistryKey builds a registry/job target_model key from a model id and
// revision -- the same "id:revision" shape helix-kernel's job contract uses.
func RegistryKey(id, revision string) string { return registry.Key(id, revision) }

// NewMock returns the deterministic, no-network embedding provider RFC-0005's
// local stack runs instead of a real model.
func NewMock(dims int) EmbeddingProvider { return providers.NewMock(dims) }

// NewEmbedder builds a guarded embedding provider for modelKey, using the
// registry's own declared dims as the expected-dimension check. Fails fast
// if the model is not registered or has no usable dims.
func NewEmbedder(reg *Registry, modelKey string, provider EmbeddingProvider, opts Options) (*Guarded, error) {
	return inference.NewEmbedder(reg, modelKey, provider, opts)
}
