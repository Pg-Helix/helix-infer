// Package usage tracks per-model call/input counts (Part 2 §4: "usage and
// cost attribution"). Phase 1 has no billing integration or persistent
// store — an in-memory counter is enough to prove the accounting hook
// exists and is wired through every provider call.
package usage

import "sync"

type Counters struct {
	Calls  int64
	Inputs int64
}

type Tracker struct {
	mu     sync.Mutex
	counts map[string]Counters
}

func NewTracker() *Tracker {
	return &Tracker{counts: make(map[string]Counters)}
}

func (t *Tracker) Record(model string, inputCount int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	c := t.counts[model]
	c.Calls++
	c.Inputs += int64(inputCount)
	t.counts[model] = c
}

// Snapshot returns a copy of the current counters, safe to read without
// racing further Record calls.
func (t *Tracker) Snapshot() map[string]Counters {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make(map[string]Counters, len(t.counts))
	for k, v := range t.counts {
		out[k] = v
	}
	return out
}
