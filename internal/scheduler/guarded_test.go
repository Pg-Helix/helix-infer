package scheduler

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Pg-Helix/helix-infer/internal/providers"
	"github.com/Pg-Helix/helix-infer/internal/usage"
)

// blockingProvider holds every call open until release is closed, so tests
// can deterministically observe how many calls are in flight at once.
type blockingProvider struct {
	inFlight atomic.Int32
	maxSeen  atomic.Int32
	release  chan struct{}
}

func (p *blockingProvider) Embed(ctx context.Context, req providers.EmbedRequest) (*providers.EmbedResult, error) {
	n := p.inFlight.Add(1)
	defer p.inFlight.Add(-1)
	for {
		seen := p.maxSeen.Load()
		if n <= seen || p.maxSeen.CompareAndSwap(seen, n) {
			break
		}
	}
	select {
	case <-p.release:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return &providers.EmbedResult{Model: req.Model, Vectors: [][]float32{{0, 0}}, Dims: 2}, nil
}

func TestGuarded_LimitsConcurrency(t *testing.T) {
	bp := &blockingProvider{release: make(chan struct{})}
	g := NewGuarded(bp, Options{MaxConcurrency: 1})

	done := make(chan struct{}, 2)
	for i := 0; i < 2; i++ {
		go func() {
			_, _ = g.Embed(context.Background(), providers.EmbedRequest{Model: "m:1", Input: []string{"x"}})
			done <- struct{}{}
		}()
	}

	time.Sleep(100 * time.Millisecond) // let both goroutines reach Embed
	if got := bp.maxSeen.Load(); got != 1 {
		t.Fatalf("max concurrent calls into the provider = %d, want 1 (MaxConcurrency=1)", got)
	}
	close(bp.release)
	<-done
	<-done
}

type failingProvider struct{ err error }

func (p *failingProvider) Embed(ctx context.Context, req providers.EmbedRequest) (*providers.EmbedResult, error) {
	return nil, p.err
}

func TestGuarded_CircuitBreakerStopsCallingAfterThreshold(t *testing.T) {
	fp := &failingProvider{err: errors.New("boom")}
	g := NewGuarded(fp, Options{MaxConcurrency: 4, CircuitFailThreshold: 2, CircuitCooldown: time.Hour})

	for i := 0; i < 2; i++ {
		_, err := g.Embed(context.Background(), providers.EmbedRequest{Model: "m:1", Input: []string{"x"}})
		if err == nil {
			t.Fatalf("call %d: expected the underlying failure, got nil", i)
		}
	}
	_, err := g.Embed(context.Background(), providers.EmbedRequest{Model: "m:1", Input: []string{"x"}})
	if !errors.Is(err, providers.ErrCircuitOpen) {
		t.Fatalf("got %v, want ErrCircuitOpen after 2 consecutive failures", err)
	}
}

type fixedDimsProvider struct{ dims int }

func (p *fixedDimsProvider) Embed(ctx context.Context, req providers.EmbedRequest) (*providers.EmbedResult, error) {
	return &providers.EmbedResult{Model: req.Model, Vectors: [][]float32{make([]float32, p.dims)}, Dims: p.dims}, nil
}

func TestGuarded_RejectsDimensionMismatch(t *testing.T) {
	g := NewGuarded(&fixedDimsProvider{dims: 5}, Options{MaxConcurrency: 1, ExpectedDims: 8})
	_, err := g.Embed(context.Background(), providers.EmbedRequest{Model: "m:1", Input: []string{"x"}})
	if !errors.Is(err, providers.ErrDimMismatch) {
		t.Fatalf("got %v, want ErrDimMismatch", err)
	}
}

func TestGuarded_AllowsMatchingDimensions(t *testing.T) {
	g := NewGuarded(&fixedDimsProvider{dims: 8}, Options{MaxConcurrency: 1, ExpectedDims: 8})
	_, err := g.Embed(context.Background(), providers.EmbedRequest{Model: "m:1", Input: []string{"x"}})
	if err != nil {
		t.Fatalf("expected no error for matching dims, got %v", err)
	}
}

func TestGuarded_RecordsUsage(t *testing.T) {
	tr := usage.NewTracker()
	g := NewGuarded(&fixedDimsProvider{dims: 2}, Options{MaxConcurrency: 1, Usage: tr})
	_, err := g.Embed(context.Background(), providers.EmbedRequest{Model: "m:1", Input: []string{"a", "b", "c"}})
	if err != nil {
		t.Fatal(err)
	}
	snap := tr.Snapshot()
	if snap["m:1"].Calls != 1 || snap["m:1"].Inputs != 3 {
		t.Fatalf("usage for m:1 = %+v, want {Calls:1 Inputs:3}", snap["m:1"])
	}
}

func TestGuarded_DoesNotRecordUsageOnFailure(t *testing.T) {
	tr := usage.NewTracker()
	g := NewGuarded(&failingProvider{err: errors.New("boom")}, Options{MaxConcurrency: 1, Usage: tr})
	_, _ = g.Embed(context.Background(), providers.EmbedRequest{Model: "m:1", Input: []string{"a"}})
	if snap := tr.Snapshot(); len(snap) != 0 {
		t.Fatalf("expected no usage recorded for a failed call, got %+v", snap)
	}
}
