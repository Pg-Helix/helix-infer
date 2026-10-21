package providers

import (
	"context"
	"reflect"
	"testing"
)

func TestMock_Conformance(t *testing.T) {
	RunConformance(t, func() EmbeddingProvider { return NewMock(8) })
}

func TestMock_DeterministicSameInput(t *testing.T) {
	m := NewMock(8)
	r1, err := m.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"hello"}})
	if err != nil {
		t.Fatal(err)
	}
	r2, err := m.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"hello"}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(r1.Vectors, r2.Vectors) {
		t.Fatalf("same input produced different vectors:\n%v\n%v", r1.Vectors, r2.Vectors)
	}
}

func TestMock_DifferentInputsDifferentVectors(t *testing.T) {
	m := NewMock(8)
	res, err := m.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"hello", "world"}})
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(res.Vectors[0], res.Vectors[1]) {
		t.Fatal("different inputs produced identical vectors")
	}
}
