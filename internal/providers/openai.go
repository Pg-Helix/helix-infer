package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"strings"
	"time"
)

// OpenAICompat calls an OpenAI-compatible /embeddings endpoint. "OpenAI-
// compatible" describes the wire shape only — it does not imply identical
// capabilities, privacy guarantees, or that this is OpenAI itself (Part 2
// §4). APIKey is a secret value held in memory only for the duration of a
// request; Embed never logs it, and any error that could otherwise echo it
// back (an HTTP client error embeds the request, a non-2xx response can
// echo request headers) is redacted before being returned.
//
// BaseURL is fixed at construction from trusted configuration, never from
// a request field or a redirect the server sends -- there is no caller-
// controlled way to retarget where the Authorization header is sent.
type OpenAICompat struct {
	BaseURL string
	APIKey  string
	// WireModel is the identifier the SERVER expects in the request body's
	// "model" field -- distinct from EmbedRequest.Model, which is cortex's
	// internal binding key ("id:revision", a registry lookup key, not a
	// wire protocol detail). Do not pass req.Model straight through: a
	// server like Ollama expects its own pulled-tag name (e.g.
	// "nomic-embed-text:v1.5"), which has no reason to match the
	// internal binding string.
	WireModel string
	// MaxRetries bounds additional attempts for a transient failure
	// (network error, 429, 5xx) within ONE Embed call -- never a second
	// durable queue; the kernel job's own lease/attempts budget remains
	// the sole source of truth for retrying a failed embedding job
	// across process restarts. 0 disables retries (a single attempt).
	MaxRetries int
	HTTPClient *http.Client
}

const (
	// maxSuccessResponseBytes bounds a 2xx response body -- an unbounded
	// decode of an arbitrarily large body from a provider is itself a
	// resource-exhaustion risk, distinct from the existing 4096-byte cap
	// on an ERROR response body.
	maxSuccessResponseBytes = 16 << 20 // 16 MiB
	maxErrorResponseBytes   = 4096
	defaultRequestTimeout   = 30 * time.Second
	defaultDialTimeout      = 5 * time.Second
)

func (p *OpenAICompat) client() *http.Client {
	if p.HTTPClient != nil {
		return p.HTTPClient
	}
	// A fresh client with explicit bounds, not http.DefaultClient (which
	// has no timeout at all -- a hung connection would block forever).
	// CheckRedirect refuses every redirect: there is no legitimate reason
	// an embeddings endpoint issues one, and following one would send
	// this request's Authorization header to whatever host the response
	// named -- a redirect-based secret-forwarding path this closes by
	// construction, not by convention.
	return &http.Client{
		Timeout: defaultRequestTimeout,
		Transport: &http.Transport{
			DialContext: (&net.Dialer{Timeout: defaultDialTimeout}).DialContext,
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return fmt.Errorf("providers: refusing to follow a redirect from the embeddings endpoint")
		},
	}
}

type openAIEmbedRequestBody struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIEmbedDatum struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type openAIEmbedResponseBody struct {
	Data  []openAIEmbedDatum `json:"data"`
	Model string             `json:"model"`
}

func (p *OpenAICompat) Embed(ctx context.Context, req EmbedRequest) (*EmbedResult, error) {
	if len(req.Input) == 0 {
		return nil, ErrEmptyInput
	}

	wireModel := p.WireModel
	if wireModel == "" {
		wireModel = req.Model
	}
	body, err := json.Marshal(openAIEmbedRequestBody{Model: wireModel, Input: req.Input})
	if err != nil {
		return nil, fmt.Errorf("providers: encode request: %w", err)
	}

	var lastErr error
	attempts := p.MaxRetries + 1
	for attempt := 0; attempt < attempts; attempt++ {
		if attempt > 0 {
			select {
			case <-time.After(time.Duration(attempt) * 200 * time.Millisecond):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		result, retryable, err := p.attemptEmbed(ctx, body, req.Input)
		if err == nil {
			return result, nil
		}
		lastErr = err
		if !retryable {
			return nil, err
		}
	}
	return nil, lastErr
}

// attemptEmbed makes one HTTP call. retryable is true only for errors
// worth a bounded retry (network failure, 429, 5xx) -- never for a 4xx
// client error or a structurally invalid response, which retrying
// cannot fix.
func (p *OpenAICompat) attemptEmbed(ctx context.Context, body []byte, input []string) (*EmbedResult, bool, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, false, p.redact(fmt.Errorf("providers: build request: %w", err))
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.APIKey)

	resp, err := p.client().Do(httpReq)
	if err != nil {
		return nil, true, p.redact(fmt.Errorf("providers: request failed: %w", err))
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorResponseBytes))
		retryable := resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500
		return nil, retryable, p.redact(fmt.Errorf("providers: openai-compatible endpoint returned %d: %s", resp.StatusCode, sanitizeForError(b)))
	}

	var parsed openAIEmbedResponseBody
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxSuccessResponseBytes)).Decode(&parsed); err != nil {
		return nil, false, p.redact(fmt.Errorf("providers: decode response: %w", err))
	}

	result, err := assembleResult(parsed, len(input))
	if err != nil {
		return nil, false, err // never retryable: the server answered, just not validly
	}
	return result, false, nil
}

// assembleResult validates the response's structure before trusting it:
// one vector per input, ordered by the response's own index field (never
// raw array position -- a provider is free to return items out of
// order), every vector the SAME dimension, every value finite, every
// vector non-degenerate (not all-zero). Any violation is
// ErrMalformedResponse, never a panic and never a partially-assembled
// result a caller could mistake for valid.
func assembleResult(parsed openAIEmbedResponseBody, wantCount int) (*EmbedResult, error) {
	if len(parsed.Data) != wantCount {
		return nil, fmt.Errorf("%w: got %d embeddings for %d inputs", ErrMalformedResponse, len(parsed.Data), wantCount)
	}
	vectors := make([][]float32, wantCount)
	seen := make([]bool, wantCount)
	dims := -1
	for _, d := range parsed.Data {
		if d.Index < 0 || d.Index >= wantCount {
			return nil, fmt.Errorf("%w: index %d out of range for %d inputs", ErrMalformedResponse, d.Index, wantCount)
		}
		if seen[d.Index] {
			return nil, fmt.Errorf("%w: duplicate index %d", ErrMalformedResponse, d.Index)
		}
		seen[d.Index] = true
		if len(d.Embedding) == 0 {
			return nil, fmt.Errorf("%w: empty embedding at index %d", ErrMalformedResponse, d.Index)
		}
		if dims == -1 {
			dims = len(d.Embedding)
		} else if len(d.Embedding) != dims {
			return nil, fmt.Errorf("%w: ragged embeddings (index %d has %d dims, expected %d)", ErrMalformedResponse, d.Index, len(d.Embedding), dims)
		}
		var sumSq float64
		for _, v := range d.Embedding {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("%w: non-finite value at index %d", ErrMalformedResponse, d.Index)
			}
			sumSq += float64(v) * float64(v)
		}
		if sumSq == 0 {
			return nil, fmt.Errorf("%w: all-zero (degenerate) embedding at index %d", ErrMalformedResponse, d.Index)
		}
		vectors[d.Index] = d.Embedding
	}
	return &EmbedResult{Model: parsed.Model, Vectors: vectors, Dims: dims}, nil
}

// sanitizeForError strips control characters from an untrusted provider
// error body before it's embedded in an error string that may reach a
// log -- bounded size (maxErrorResponseBytes) is already enforced by the
// caller's LimitReader; this additionally keeps a malicious body from
// injecting terminal escapes/newlines into log output.
func sanitizeForError(b []byte) string {
	return strings.Map(func(r rune) rune {
		if r < 0x20 && r != '\n' && r != '\t' {
			return -1
		}
		return r
	}, string(b))
}

// redact replaces any occurrence of the API key with a fixed placeholder
// before an error ever leaves this adapter — credentials never reach logs,
// traces or SQL results through an error value (Part 2 §4).
func (p *OpenAICompat) redact(err error) error {
	if p.APIKey == "" || err == nil {
		return err
	}
	return fmt.Errorf("%s", strings.ReplaceAll(err.Error(), p.APIKey, "[REDACTED]"))
}
