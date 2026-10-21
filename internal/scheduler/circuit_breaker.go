package scheduler

import (
	"sync"
	"time"
)

// circuitBreaker trips after failThreshold consecutive failures and stays
// open for cooldown before allowing another attempt. A single successful
// call (including the first attempt after cooldown) resets it.
type circuitBreaker struct {
	mu               sync.Mutex
	failThreshold    int
	cooldown         time.Duration
	consecutiveFails int
	openUntil        time.Time
}

func newCircuitBreaker(failThreshold int, cooldown time.Duration) *circuitBreaker {
	return &circuitBreaker{failThreshold: failThreshold, cooldown: cooldown}
}

func (b *circuitBreaker) Allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.openUntil.IsZero() || !time.Now().Before(b.openUntil)
}

func (b *circuitBreaker) Record(err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err == nil {
		b.consecutiveFails = 0
		b.openUntil = time.Time{}
		return
	}
	b.consecutiveFails++
	if b.consecutiveFails >= b.failThreshold {
		b.openUntil = time.Now().Add(b.cooldown)
	}
}
