package scheduler

import (
	"errors"
	"testing"
	"time"
)

func TestCircuitBreaker_TripsAfterConsecutiveFailures(t *testing.T) {
	b := newCircuitBreaker(3, 50*time.Millisecond)
	for i := 0; i < 2; i++ {
		if !b.Allow() {
			t.Fatalf("breaker open too early at failure %d", i)
		}
		b.Record(errors.New("boom"))
	}
	if !b.Allow() {
		t.Fatal("breaker open before reaching the failure threshold")
	}
	b.Record(errors.New("boom")) // 3rd consecutive failure: trips
	if b.Allow() {
		t.Fatal("breaker should be open after 3 consecutive failures")
	}
}

func TestCircuitBreaker_ResetsOnSuccess(t *testing.T) {
	b := newCircuitBreaker(2, time.Second)
	b.Record(errors.New("boom"))
	b.Record(nil) // success resets the streak
	b.Record(errors.New("boom"))
	if !b.Allow() {
		t.Fatal("breaker should still be closed: only 1 consecutive failure since the reset")
	}
}

func TestCircuitBreaker_RecoversAfterCooldown(t *testing.T) {
	b := newCircuitBreaker(1, 30*time.Millisecond)
	b.Record(errors.New("boom"))
	if b.Allow() {
		t.Fatal("breaker should be open immediately after tripping")
	}
	time.Sleep(50 * time.Millisecond)
	if !b.Allow() {
		t.Fatal("breaker should allow attempts again after the cooldown elapses")
	}
}
