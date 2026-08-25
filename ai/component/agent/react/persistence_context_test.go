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
	"testing"
	"time"

	"dubbo-admin-ai/component/agent/fallback"
	"dubbo-admin-ai/schema"
	conversationstore "dubbo-admin-ai/store"
	memorystore "dubbo-admin-ai/store/memory"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type recordingStore struct {
	conversationstore.Store
	addContexts  []context.Context
	nextContexts []context.Context
	nextContext  chan context.Context
}

func (s *recordingStore) AddHistory(ctx context.Context, sessionID string, messages ...*ai.Message) error {
	s.addContexts = append(s.addContexts, ctx)
	return s.Store.AddHistory(ctx, sessionID, messages...)
}

func (s *recordingStore) NextTurn(ctx context.Context, sessionID string) error {
	s.nextContexts = append(s.nextContexts, ctx)
	err := s.Store.NextTurn(ctx, sessionID)
	if err == nil && s.nextContext != nil {
		s.nextContext <- ctx
	}
	return err
}

func TestGeneratedMessageUsesDetachedPersistenceContext(t *testing.T) {
	backing := memorystore.NewMemoryStore(2)
	now := time.Now()
	if err := backing.Create(context.Background(), &conversationstore.Session{
		ID:        "persist-context-session",
		CreatedAt: now,
		UpdatedAt: now,
		Status:    "active",
	}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	store := &recordingStore{Store: backing}
	ra := &ReActAgent{
		registry:     genkit.Init(context.Background()),
		messageStore: store,
		fallback:     fallback.NewHandler(),
	}

	requestCtx, cancel := context.WithCancel(context.Background())
	ctx, state, err := ra.newInteraction(requestCtx, &schema.UserInput{Content: "hello"}, "persist-context-session")
	if err != nil {
		t.Fatalf("newInteraction() error = %v", err)
	}
	if state.persistCtx == nil || state.persistCtx.Err() != nil {
		t.Fatalf("persistCtx = %v, want a non-cancelable context", state.persistCtx)
	}

	if _, err := ra.reasonActStep(&stubPrompt{resp: textResp("answer")}, nil, 0)(ctx, state); err != nil {
		t.Fatalf("reasonActStep() error = %v", err)
	}
	if len(store.addContexts) != 2 {
		t.Fatalf("AddHistory() calls = %d, want user and generated messages", len(store.addContexts))
	}

	cancel()
	if requestCtx.Err() == nil {
		t.Fatal("request context was not canceled")
	}
	if store.addContexts[1].Err() != nil {
		t.Fatalf("generated message context error = %v, want nil", store.addContexts[1].Err())
	}
}

func TestInteractionFinalizationUsesDetachedContext(t *testing.T) {
	backing := memorystore.NewMemoryStore(1)
	now := time.Now()
	if err := backing.Create(context.Background(), &conversationstore.Session{
		ID:        "finalize-context-session",
		CreatedAt: now,
		UpdatedAt: now,
		Status:    "active",
	}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	nextContext := make(chan context.Context, 1)
	store := &recordingStore{Store: backing, nextContext: nextContext}
	ra := &ReActAgent{messageStore: store, bufferSize: 1}
	requestCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ra.Interact(requestCtx, &schema.UserInput{Content: "hello"}, "finalize-context-session")

	var persistCtx context.Context
	select {
	case persistCtx = <-nextContext:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for NextTurn")
	}
	cancel()
	if requestCtx.Err() == nil {
		t.Fatal("request context was not canceled")
	}
	if persistCtx.Err() != nil {
		t.Fatalf("NextTurn context error = %v, want nil after request cancellation", persistCtx.Err())
	}
}
