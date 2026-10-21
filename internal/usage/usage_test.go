package usage

import (
	"sync"
	"testing"
)

func TestTracker_RecordAndSnapshot(t *testing.T) {
	tr := NewTracker()
	tr.Record("m:1", 3)
	tr.Record("m:1", 2)
	tr.Record("m:2", 1)

	snap := tr.Snapshot()
	if snap["m:1"].Calls != 2 || snap["m:1"].Inputs != 5 {
		t.Fatalf("m:1 = %+v, want {Calls:2 Inputs:5}", snap["m:1"])
	}
	if snap["m:2"].Calls != 1 || snap["m:2"].Inputs != 1 {
		t.Fatalf("m:2 = %+v, want {Calls:1 Inputs:1}", snap["m:2"])
	}
}

func TestTracker_ConcurrentRecord(t *testing.T) {
	tr := NewTracker()
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tr.Record("m:1", 1)
		}()
	}
	wg.Wait()
	if got := tr.Snapshot()["m:1"].Calls; got != 100 {
		t.Fatalf("Calls = %d, want 100", got)
	}
}
