package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OpenAICompat calls an OpenAI-compatible /embeddings endpoint. "OpenAI-
// compatible" describes the wire shape only — it does not imply identical
// capabilities, privacy guarantees, or that this is OpenAI itself (Part 2
// §4). APIKey is a secret value held in memory only for the duration of a
// request; Embed never logs it, and any error that could otherwise echo it
// back (an HTTP client error embeds the request, a non-2xx response can
// echo request headers) is redacted before being returned.
type OpenAICompat struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

func (p *OpenAICompat) client() *http.Client {
	if p.HTTPClient != nil {
		return p.HTTPClient
	}
	return http.DefaultClient
}

type openAIEmbedRequestBody struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIEmbedResponseBody struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (p *OpenAICompat) Embed(ctx context.Context, req EmbedRequest) (*EmbedResult, error) {
	if len(req.Input) == 0 {
		return nil, ErrEmptyInput
	}

	body, err := json.Marshal(openAIEmbedRequestBody{Model: req.Model, Input: req.Input})
	if err != nil {
		return nil, fmt.Errorf("providers: encode request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, p.redact(fmt.Errorf("providers: build request: %w", err))
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.APIKey)

	resp, err := p.client().Do(httpReq)
	if err != nil {
		return nil, p.redact(fmt.Errorf("providers: request failed: %w", err))
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, p.redact(fmt.Errorf("providers: openai-compatible endpoint returned %d: %s", resp.StatusCode, b))
	}

	var parsed openAIEmbedResponseBody
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, p.redact(fmt.Errorf("providers: decode response: %w", err))
	}

	vectors := make([][]float32, len(parsed.Data))
	dims := 0
	for i, d := range parsed.Data {
		vectors[i] = d.Embedding
		if len(d.Embedding) > dims {
			dims = len(d.Embedding)
		}
	}
	return &EmbedResult{Model: req.Model, Vectors: vectors, Dims: dims}, nil
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
