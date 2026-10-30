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

// OpenAICompatConfig configures a real OpenAI-compatible embeddings
// provider (e.g. Ollama's /v1/embeddings). WireModel is the identifier
// the SERVER expects (its own pulled model/tag name), which is NOT the
// same string as the internal "id:revision" binding key callers pass as
// EmbedRequest.Model -- see providers.OpenAICompat's own doc comment.
type OpenAICompatConfig struct {
	BaseURL    string
	APIKey     string
	WireModel  string
	MaxRetries int
}

// NewOpenAICompat returns a real, network-calling embedding provider
// against an OpenAI-compatible /embeddings endpoint. The only
// constructor in this module for that adapter -- callers (helix-cortex)
// must not copy providers.OpenAICompat's logic locally; cmd/cortex and
// cmd/cortex-api both import this package for exactly that reason, so
// the two Workflow 2 processes (chunk-worker, cortex-api's query
// embedder) share one validated implementation.
func NewOpenAICompat(cfg OpenAICompatConfig) EmbeddingProvider {
	return &providers.OpenAICompat{
		BaseURL:    cfg.BaseURL,
		APIKey:     cfg.APIKey,
		WireModel:  cfg.WireModel,
		MaxRetries: cfg.MaxRetries,
	}
}

// NewEmbedder builds a guarded embedding provider for modelKey, using the
// registry's own declared dims as the expected-dimension check. Fails fast
// if the model is not registered or has no usable dims.
func NewEmbedder(reg *Registry, modelKey string, provider EmbeddingProvider, opts Options) (*Guarded, error) {
	return inference.NewEmbedder(reg, modelKey, provider, opts)
}
