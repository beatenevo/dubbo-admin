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
	"fmt"

	"dubbo-admin-ai/component/agent"
	"dubbo-admin-ai/component/agent/fallback"
	"dubbo-admin-ai/schema"
	conversationstore "dubbo-admin-ai/store"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type contextKey string

const sessionIDContextKey contextKey = "session"

// ReActAgent is a ReAct-strategy agent: each interaction runs the configured
// stages (reasonAct then observe) in a bounded loop until the observe stage
// produces a final answer or the iteration budget is exhausted. A single agent
// is safe for concurrent interactions — per-interaction state lives in the
// Channels/state returned by Interact, not on the agent.
type ReActAgent struct {
	registry     *genkit.Genkit
	messageStore conversationstore.MessageStore
	fallback     *fallback.Handler // single shared fallback handler

	stages       []builtStage
	toolTimeouts toolTimeoutResolver

	defaultModel   string // Default model in "provider/model" format (e.g., "dashscope/qwen-max")
	promptBasePath string
	maxIterations  int
	bufferSize     int
}

// NewReActAgent builds a ReActAgent, assembling one prompt per configured stage
// up front so the per-interaction hot path only executes them. It returns an
// error if any stage's prompt file is missing or a stage is misconfigured.
func NewReActAgent(g *genkit.Genkit, messageStore conversationstore.MessageStore, promptBasePath string, defaultModel string, maxIterations int, stageChannelBufferSize int, stagesCfg []StageInfo, toolTimeouts toolTimeoutResolver, toolRefs []ai.ToolRef) (*ReActAgent, error) {
	if messageStore == nil {
		return nil, fmt.Errorf("message store is nil")
	}
	ra := &ReActAgent{
		registry:       g,
		messageStore:   messageStore,
		fallback:       fallback.NewHandler(),
		toolTimeouts:   toolTimeouts,
		defaultModel:   defaultModel,
		promptBasePath: promptBasePath,
		maxIterations:  maxIterations,
		bufferSize:     max(stageChannelBufferSize, 1),
	}

	stages, err := ra.buildStages(g, stagesCfg, promptBasePath, defaultModel, toolRefs)
	if err != nil {
		return nil, err
	}
	ra.stages = stages
	return ra, nil
}

// Interact runs one interaction asynchronously and returns immediately with the
// Channels the caller streams from. The loop, final answer emission, and channel
// close all happen on a background goroutine; the caller owns draining Channels.
func (ra *ReActAgent) Interact(parent context.Context, input *schema.UserInput, sessionID string) *agent.Channels {
	chans := agent.NewChannels(ra.bufferSize)
	go func() {
		if parent == nil {
			parent = context.Background()
		}
		ctx, s, err := ra.newInteraction(parent, input, sessionID)
		if err != nil {
			chans.ErrorChan <- err
			chans.Close()
			return
		}

		if err := runLoop(ctx, s, ra.maxIterations, ra.buildSteps(chans)...); err != nil {
			chans.ErrorChan <- err
			chans.Close()
			return
		}

		if err := ra.messageStore.NextTurn(ctx, sessionID); err != nil {
			chans.ErrorChan <- fmt.Errorf("failed to complete turn: %w", err)
			chans.Close()
			return
		}

		// Emit the final answer for the SSE layer; it carries the accumulated
		// usage that the MessageDelta needs. Fall back to an empty observation
		// if the loop produced none (e.g. error before observe ran).
		final := s.Observe
		if final == nil {
			final = &schema.Observation{UsageInfo: s.Usage}
		}
		chans.Send(schema.StreamFinal(final))

		chans.Close()
	}()
	return chans
}

// newInteraction records the user input into history and returns a session-scoped
// context plus a fresh state.
func (ra *ReActAgent) newInteraction(parent context.Context, input *schema.UserInput, sessionID string) (context.Context, *state, error) {
	if ra.messageStore == nil {
		return nil, nil, fmt.Errorf("message store is not configured")
	}
	if input == nil {
		return nil, nil, fmt.Errorf("user input is nil")
	}

	// Record the user's message as plain text. The session id travels via
	// context, so there is no need to wrap the input in a
	// JSON envelope the model would otherwise have to read through.
	if err := ra.messageStore.AddHistory(parent, sessionID, ai.NewUserMessage(ai.NewTextPart(input.Content))); err != nil {
		return nil, nil, fmt.Errorf("failed to record user message: %w", err)
	}

	ctx := context.WithValue(parent, sessionIDContextKey, sessionID)
	ctx = withCurrentPageContext(ctx, input.Context)
	s := &state{Input: input, Session: sessionID, Usage: &ai.GenerationUsage{}}
	return ctx, s, nil
}
