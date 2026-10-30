package providers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func newTestServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

func TestOpenAICompat_Conformance(t *testing.T) {
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req openAIEmbedRequestBody
		_ = json.NewDecoder(r.Body).Decode(&req)
		resp := openAIEmbedResponseBody{}
		for i := range req.Input {
			resp.Data = append(resp.Data, openAIEmbedDatum{Embedding: []float32{0.1, 0.2, 0.3}, Index: i})
		}
		_ = json.NewEncoder(w).Encode(resp)
	})
	RunConformance(t, func() EmbeddingProvider {
		return &OpenAICompat{BaseURL: srv.URL, APIKey: "test-key"}
	})
}

func TestOpenAICompat_ParsesEmbeddings(t *testing.T) {
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(openAIEmbedResponseBody{Data: []openAIEmbedDatum{
			{Embedding: []float32{1, 2, 3}, Index: 0},
			{Embedding: []float32{4, 5, 6}, Index: 1},
		}})
	})
	p := &OpenAICompat{BaseURL: srv.URL, APIKey: "test-key"}
	res, err := p.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"a", "b"}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Dims != 3 || len(res.Vectors) != 2 {
		t.Fatalf("got dims=%d vectors=%d, want dims=3 vectors=2", res.Dims, len(res.Vectors))
	}
}

// TestOpenAICompat_ReordersByIndex proves a provider that answers out of
// array order (but with correct index fields) is still assembled
// correctly -- raw array position is never trusted.
func TestOpenAICompat_ReordersByIndex(t *testing.T) {
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(openAIEmbedResponseBody{Data: []openAIEmbedDatum{
			{Embedding: []float32{9, 9, 9}, Index: 1}, // "b"'s vector, listed first
			{Embedding: []float32{1, 1, 1}, Index: 0}, // "a"'s vector, listed second
		}})
	})
	p := &OpenAICompat{BaseURL: srv.URL, APIKey: "test-key"}
	res, err := p.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"a", "b"}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Vectors[0][0] != 1 || res.Vectors[1][0] != 9 {
		t.Fatalf("got vectors in wrong order: %v", res.Vectors)
	}
}

func TestOpenAICompat_RejectsMalformedResponses(t *testing.T) {
	cases := []struct {
		name string
		data []openAIEmbedDatum
	}{
		{"wrong count", []openAIEmbedDatum{{Embedding: []float32{1, 2}, Index: 0}}},
		{"duplicate index", []openAIEmbedDatum{{Embedding: []float32{1, 2}, Index: 0}, {Embedding: []float32{3, 4}, Index: 0}}},
		{"out of range index", []openAIEmbedDatum{{Embedding: []float32{1, 2}, Index: 0}, {Embedding: []float32{3, 4}, Index: 5}}},
		{"empty vector", []openAIEmbedDatum{{Embedding: []float32{1, 2}, Index: 0}, {Embedding: []float32{}, Index: 1}}},
		{"ragged vectors", []openAIEmbedDatum{{Embedding: []float32{1, 2}, Index: 0}, {Embedding: []float32{3, 4, 5}, Index: 1}}},
		{"all-zero vector", []openAIEmbedDatum{{Embedding: []float32{1, 2}, Index: 0}, {Embedding: []float32{0, 0}, Index: 1}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewEncoder(w).Encode(openAIEmbedResponseBody{Data: c.data})
			})
			p := &OpenAICompat{BaseURL: srv.URL, APIKey: "test-key"}
			_, err := p.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"a", "b"}})
			if err == nil {
				t.Fatalf("expected ErrMalformedResponse, got nil")
			}
			if !errors.Is(err, ErrMalformedResponse) {
				t.Fatalf("expected ErrMalformedResponse, got %v", err)
			}
		})
	}
}

// A NaN/Inf value cannot actually reach assembleResult's own finite
// check through this adapter's encoding/json-based decode: Go's decoder
// itself refuses to unmarshal a numeric literal that overflows float32
// (confirmed directly -- it errors at decode time, before assembleResult
// ever runs), and NaN/Infinity are not valid JSON number tokens at all.
// That decode error is still treated as non-retryable, so the safety
// property holds either way; assembleResult's own check is kept as a
// defensive invariant on the validated-vector contract itself, not
// because this specific decode path can trigger it.

// TestOpenAICompat_WireModelMapping proves the internal binding key
// (EmbedRequest.Model) is never sent as-is when WireModel names a
// different server-side identifier.
func TestOpenAICompat_WireModelMapping(t *testing.T) {
	var gotModel string
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req openAIEmbedRequestBody
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotModel = req.Model
		_ = json.NewEncoder(w).Encode(openAIEmbedResponseBody{Data: []openAIEmbedDatum{{Embedding: []float32{1}, Index: 0}}})
	})
	p := &OpenAICompat{BaseURL: srv.URL, APIKey: "test-key", WireModel: "nomic-embed-text:v1.5"}
	_, err := p.Embed(context.Background(), EmbedRequest{Model: "local-real-embed:v1", Input: []string{"a"}})
	if err != nil {
		t.Fatal(err)
	}
	if gotModel != "nomic-embed-text:v1.5" {
		t.Fatalf("got wire model %q, want %q (the internal binding key must not be sent as-is)", gotModel, "nomic-embed-text:v1.5")
	}
}

// TestOpenAICompat_RetriesTransientFailures proves a 5xx is retried up
// to MaxRetries times and succeeds once the server recovers, while a
// 4xx is never retried.
func TestOpenAICompat_RetriesTransientFailures(t *testing.T) {
	var calls int
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_ = json.NewEncoder(w).Encode(openAIEmbedResponseBody{Data: []openAIEmbedDatum{{Embedding: []float32{1}, Index: 0}}})
	})
	p := &OpenAICompat{BaseURL: srv.URL, APIKey: "test-key", MaxRetries: 2}
	_, err := p.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"a"}})
	if err != nil {
		t.Fatalf("expected eventual success after retries, got %v", err)
	}
	if calls != 3 {
		t.Fatalf("got %d calls, want exactly 3 (2 failures + 1 success)", calls)
	}
}

func TestOpenAICompat_DoesNotRetryClientErrors(t *testing.T) {
	var calls int
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusBadRequest)
	})
	p := &OpenAICompat{BaseURL: srv.URL, APIKey: "test-key", MaxRetries: 2}
	_, err := p.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"a"}})
	if err == nil {
		t.Fatal("expected an error")
	}
	if calls != 1 {
		t.Fatalf("got %d calls, want exactly 1 (a 4xx must not be retried)", calls)
	}
}

// TestOpenAICompat_RefusesRedirects proves a redirect from the
// embeddings endpoint is never followed -- following one would send
// this request's Authorization header to whatever host the response
// named.
func TestOpenAICompat_RefusesRedirects(t *testing.T) {
	var secondHostHit bool
	second := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		secondHostHit = true
	})
	first := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, second.URL+"/embeddings", http.StatusFound)
	})
	p := &OpenAICompat{BaseURL: first.URL, APIKey: "test-key"}
	_, err := p.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"a"}})
	if err == nil {
		t.Fatal("expected an error refusing the redirect")
	}
	if secondHostHit {
		t.Fatal("the redirect was followed -- the second host must never be contacted")
	}
}

func TestOpenAICompat_NonSuccessStatus(t *testing.T) {
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limited"}`))
	})
	p := &OpenAICompat{BaseURL: srv.URL, APIKey: "test-key"}
	_, err := p.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"a"}})
	if err == nil || !strings.Contains(err.Error(), "429") {
		t.Fatalf("expected an error mentioning status 429, got %v", err)
	}
}

// TestOpenAICompat_RedactsCredential proves the API key never survives into
// a returned error, even when the upstream response itself echoes back the
// Authorization header value (e.g. a misconfigured gateway's error page
// reflecting request headers) — Embed must still scrub it before returning.
func TestOpenAICompat_RedactsCredential(t *testing.T) {
	const secret = "sk-super-secret-value-9f1e"
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte("upstream rejected Authorization: " + r.Header.Get("Authorization")))
	})
	p := &OpenAICompat{BaseURL: srv.URL, APIKey: secret}
	_, err := p.Embed(context.Background(), EmbedRequest{Model: "m:1", Input: []string{"a"}})
	if err == nil {
		t.Fatal("expected an error")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("API key leaked into error: %v", err)
	}
	if !strings.Contains(err.Error(), "[REDACTED]") {
		t.Fatalf("expected a redaction placeholder in error: %v", err)
	}
}

func TestOpenAICompat_RespectsClientTimeout(t *testing.T) {
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(2 * time.Second):
		case <-r.Context().Done():
		}
	})
	p := &OpenAICompat{BaseURL: srv.URL, APIKey: "test-key"}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := p.Embed(ctx, EmbedRequest{Model: "m:1", Input: []string{"a"}})
	if err == nil {
		t.Fatal("expected a timeout error")
	}
	if elapsed := time.Since(start); elapsed > 1*time.Second {
		t.Fatalf("Embed took %v, did not respect the context timeout", elapsed)
	}
}

func TestOpenAICompat_RejectsEmptyInput(t *testing.T) {
	p := &OpenAICompat{BaseURL: "http://unused.invalid", APIKey: "k"}
	_, err := p.Embed(context.Background(), EmbedRequest{Model: "m:1"})
	if err != ErrEmptyInput {
		t.Fatalf("got %v, want ErrEmptyInput", err)
	}
}
