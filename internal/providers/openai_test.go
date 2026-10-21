package providers

import (
	"context"
	"encoding/json"
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
		for range req.Input {
			resp.Data = append(resp.Data, struct {
				Embedding []float32 `json:"embedding"`
			}{Embedding: []float32{0.1, 0.2, 0.3}})
		}
		_ = json.NewEncoder(w).Encode(resp)
	})
	RunConformance(t, func() EmbeddingProvider {
		return &OpenAICompat{BaseURL: srv.URL, APIKey: "test-key"}
	})
}

func TestOpenAICompat_ParsesEmbeddings(t *testing.T) {
	srv := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(openAIEmbedResponseBody{Data: []struct {
			Embedding []float32 `json:"embedding"`
		}{
			{Embedding: []float32{1, 2, 3}},
			{Embedding: []float32{4, 5, 6}},
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
