/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package react

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"dubbo-admin-ai/component/agent/fallback"
	"dubbo-admin-ai/schema"
	conversationstore "dubbo-admin-ai/store"
	memorystore "dubbo-admin-ai/store/memory"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// stubPrompt is an ai.Prompt whose Execute returns a canned response/error,
// letting the step unit tests run without a live model.
type stubPrompt struct {
	resp *ai.ModelResponse
	err  error
}

func (s *stubPrompt) Name() string { return "stub" }
func (s *stubPrompt) Execute(ctx context.Context, opts ...ai.PromptExecuteOption) (*ai.ModelResponse, error) {
	return s.resp, s.err
}
func (s *stubPrompt) Render(ctx context.Context, input any) (*ai.GenerateActionOptions, error) {
	return &ai.GenerateActionOptions{}, nil
}

func contextWithHistory(s conversationstore.Store, sessionID string) (context.Context, uint64) {
	now := time.Now()
	if err := s.Create(context.Background(), &conversationstore.Session{
		ID:        sessionID,
		CreatedAt: now,
		UpdatedAt: now,
		Status:    "active",
	}); err != nil {
		panic(err)
	}
	turnID, err := s.BeginTurn(context.Background(), sessionID)
	if err != nil {
		panic(err)
	}
	if err := s.AddHistoryToTurn(context.Background(), sessionID, turnID, ai.NewUserMessage(ai.NewTextPart("hello"))); err != nil {
		panic(err)
	}
	return context.WithValue(context.Background(), sessionIDContextKey, sessionID), turnID
}

func testAgent(g *genkit.Genkit) (*ReActAgent, conversationstore.Store) {
	s := memorystore.NewMemoryStore()
	return &ReActAgent{registry: g, messageStore: s, fallback: fallback.NewHandler()}, s
}

func textResp(text string) *ai.ModelResponse {
	return &ai.ModelResponse{Message: ai.NewMessage(ai.RoleModel, nil, ai.NewTextPart(text))}
}

func toolReqResp(name string, input map[string]any) *ai.ModelResponse {
	return &ai.ModelResponse{Message: ai.NewMessage(ai.RoleModel, nil,
		ai.NewToolRequestPart(&ai.ToolRequest{Name: name, Input: input}),
	)}
}

func TestReasonActStep(t *testing.T) {
	tests := []struct {
		name       string
		setup      func(*genkit.Genkit) ai.Prompt
		errContain string
		assertFn   func(t *testing.T, s *state)
	}{
		{
			name:  "no_tool_call_answers_directly",
			setup: func(*genkit.Genkit) ai.Prompt { return &stubPrompt{resp: textResp("Dubbo is an RPC framework.")} },
			assertFn: func(t *testing.T, s *state) {
				if s.Tools == nil || len(s.Tools.Outputs) != 0 {
					t.Fatalf("expected empty tool outputs, got %+v", s.Tools)
				}
			},
		},
		{
			name: "with_tool_call_returns_outputs",
			setup: func(g *genkit.Genkit) ai.Prompt {
				genkit.DefineTool(g, "mock_tool", "mock tool", func(ctx *ai.ToolContext, input map[string]any) (map[string]any, error) {
					return map[string]any{"tool_name": "mock_tool", "summary": "ok", "result": map[string]any{"echo": input["q"]}}, nil
				})
				return &stubPrompt{resp: toolReqResp("mock_tool", map[string]any{"q": "ping"})}
			},
			assertFn: func(t *testing.T, s *state) {
				if s.Tools == nil || len(s.Tools.Outputs) < 1 || s.Tools.Outputs[0].ToolName != "mock_tool" {
					t.Fatalf("unexpected tool outputs: %+v", s.Tools)
				}
			},
		},
		{
			// A failing tool must not abort the interaction: the step records the
			// failure as a tool output (so observe can degrade) and returns no error.
			name: "tool_error_degrades_not_aborts",
			setup: func(g *genkit.Genkit) ai.Prompt {
				genkit.DefineTool(g, "broken_tool", "broken", func(ctx *ai.ToolContext, input map[string]any) (map[string]any, error) {
					return nil, errors.New("boom")
				})
				return &stubPrompt{resp: toolReqResp("broken_tool", map[string]any{"x": 1})}
			},
			assertFn: func(t *testing.T, s *state) {
				if s.Tools == nil || len(s.Tools.Outputs) != 1 {
					t.Fatalf("expected one recorded tool output, got %+v", s.Tools)
				}
				out := s.Tools.Outputs[0]
				if out.ToolName != "broken_tool" || !strings.Contains(out.Summary, "failed") {
					t.Fatalf("expected degraded failure output for broken_tool, got %+v", out)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := genkit.Init(context.Background())
			ra, backingStore := testAgent(g)
			s := &state{Input: &schema.UserInput{Content: "hi"}, Session: "session", Usage: &ai.GenerationUsage{}}

			ctx, turnID := contextWithHistory(backingStore, "session")
			s.TurnID = turnID
			done, err := ra.reasonActStep(tt.setup(g), nil, 0)(ctx, s)
			if tt.errContain != "" {
				if err == nil || !strings.Contains(err.Error(), tt.errContain) {
					t.Fatalf("expected error containing %q, got %v", tt.errContain, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("reasonAct step error: %v", err)
			}
			if done {
				t.Fatalf("reasonAct step should never terminate the loop")
			}
			tt.assertFn(t, s)
		})
	}
}

func TestReasonActStepExecuteError(t *testing.T) {
	g := genkit.Init(context.Background())
	ra, backingStore := testAgent(g)
	prompt := &stubPrompt{resp: nil, err: errors.New("execute failed")}
	s := &state{Input: &schema.UserInput{Content: "hi"}, Session: "s3", Usage: &ai.GenerationUsage{}}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("unexpected panic: %v", r)
		}
	}()

	ctx, turnID := contextWithHistory(backingStore, "s3")
	s.TurnID = turnID
	_, err := ra.reasonActStep(prompt, nil, 0)(ctx, s)
	if err == nil || !strings.Contains(err.Error(), "failed to execute reasonAct prompt") {
		t.Fatalf("expected wrapped execute error, got %v", err)
	}
}
