// Package scheduler wraps an EmbeddingProvider with the operational
// guarantees Part 2 §4 asks for around any provider call: a concurrency
// limit, a circuit breaker, usage accounting, and an embedding-dimension
// check against the registry. None of this is provider-specific, so it
// wraps any providers.EmbeddingProvider rather than living inside one
// adapter.
package scheduler

import (
	"context"
	"fmt"
	"time"

	"github.com/Pg-Helix/helix-infer/internal/providers"
	"github.com/Pg-Helix/helix-infer/internal/usage"
)

// Options configures a Guarded provider. Zero values are usable defaults
// except MaxConcurrency, which must be positive (a zero-size semaphore
// channel would block every call forever).
type Options struct {
	MaxConcurrency       int
	CircuitFailThreshold int // consecutive failures before the breaker opens; 0 disables it
	CircuitCooldown      time.Duration
	ExpectedDims         int // 0 disables the dimension check
	Usage                *usage.Tracker
}

type Guarded struct {
	inner   providers.EmbeddingProvider
	sem     chan struct{}
	breaker *circuitBreaker
	opts    Options
}

func NewGuarded(inner providers.EmbeddingProvider, opts Options) *Guarded {
	if opts.MaxConcurrency <= 0 {
		opts.MaxConcurrency = 1
	}
	failThreshold := opts.CircuitFailThreshold
	if failThreshold <= 0 {
		failThreshold = 1 << 30 // effectively disabled: never trips
	}
	return &Guarded{
		inner:   inner,
		sem:     make(chan struct{}, opts.MaxConcurrency),
		breaker: newCircuitBreaker(failThreshold, opts.CircuitCooldown),
		opts:    opts,
	}
}

func (g *Guarded) Embed(ctx context.Context, req providers.EmbedRequest) (*providers.EmbedResult, error) {
	if !g.breaker.Allow() {
		return nil, providers.ErrCircuitOpen
	}

	select {
	case g.sem <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-g.sem }()

	res, err := g.inner.Embed(ctx, req)
	g.breaker.Record(err)
	if err != nil {
		return nil, err
	}

	if g.opts.ExpectedDims > 0 && res.Dims != g.opts.ExpectedDims {
		return nil, fmt.Errorf("%w: got %d, want %d", providers.ErrDimMismatch, res.Dims, g.opts.ExpectedDims)
	}

	if g.opts.Usage != nil {
		g.opts.Usage.Record(req.Model, len(req.Input))
	}
	return res, nil
}
