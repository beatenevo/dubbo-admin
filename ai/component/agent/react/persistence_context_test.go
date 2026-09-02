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
	"testing"
	"time"

	"dubbo-admin-ai/component/agent/fallback"
	"dubbo-admin-ai/schema"
	conversationstore "dubbo-admin-ai/store"
	memorystore "dubbo-admin-ai/store/memory"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

type failingUserWriteStore struct {
	conversationstore.Store
	abortCalled chan struct{}
}

func (s *failingUserWriteStore) AddHistoryToTurn(context.Context, string, uint64, ...*ai.Message) error {
	return errors.New("user message write failed")
}

func (s *failingUserWriteStore) AbortTurnForTurn(ctx context.Context, sessionID string, turnID uint64) error {
	select {
	case <-s.abortCalled:
	default:
		close(s.abortCalled)
	}
	return s.Store.AbortTurnForTurn(ctx, sessionID, turnID)
}

type recordingStore struct {
	conversationstore.Store
	addContexts  []context.Context
	nextContexts []context.Context
	nextContext  chan context.Context
	nextRelease  chan struct{}
}

func (s *recordingStore) AddHistoryToTurn(ctx context.Context, sessionID string, turnID uint64, messages ...*ai.Message) error {
	s.addContexts = append(s.addContexts, ctx)
	return s.Store.AddHistoryToTurn(ctx, sessionID, turnID, messages...)
}

func (s *recordingStore) NextTurnForTurn(ctx context.Context, sessionID string, turnID uint64) error {
	s.nextContexts = append(s.nextContexts, ctx)
	if s.nextContext != nil {
		s.nextContext <- ctx
	}
	if s.nextRelease != nil {
		<-s.nextRelease
	}
	return s.Store.NextTurnForTurn(ctx, sessionID, turnID)
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
	defer state.cancelPersistence()
	if state.persistCtx == nil || state.persistCtx.Err() != nil {
		t.Fatalf("persistCtx = %v, want an active context", state.persistCtx)
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

func TestUserWriteFailureAbortsStartedTurn(t *testing.T) {
	backing := memorystore.NewMemoryStore(1)
	now := time.Now()
	if err := backing.Create(context.Background(), &conversationstore.Session{
		ID:        "abort-user-write-session",
		CreatedAt: now,
		UpdatedAt: now,
		Status:    "active",
	}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	store := &failingUserWriteStore{Store: backing, abortCalled: make(chan struct{})}
	ra := &ReActAgent{messageStore: store}
	if _, _, err := ra.newInteraction(context.Background(), &schema.UserInput{Content: "hello"}, "abort-user-write-session"); err == nil {
		t.Fatal("newInteraction() error = nil, want user write failure")
	}
	select {
	case <-store.abortCalled:
	case <-time.After(time.Second):
		t.Fatal("AbortTurnForTurn was not called")
	}
	if messages, err := backing.AllMemory(context.Background(), "abort-user-write-session"); err != nil || len(messages) != 0 {
		t.Fatalf("AllMemory() after failed user write = %#v, error = %v, want empty history", messages, err)
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
	nextRelease := make(chan struct{})
	store := &recordingStore{Store: backing, nextContext: nextContext, nextRelease: nextRelease}
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
		t.Fatalf("NextTurn context error = %v, want active after request cancellation", persistCtx.Err())
	}
	close(nextRelease)

	deadline, ok := persistCtx.Deadline()
	if !ok {
		t.Fatal("NextTurn context has no deadline")
	}
	if remaining := time.Until(deadline); remaining <= 0 || remaining > persistenceTimeout {
		t.Fatalf("persistence deadline remaining = %v, want within %v", remaining, persistenceTimeout)
	}

	select {
	case <-persistCtx.Done():
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for interaction completion")
	}
	if persistCtx.Err() != context.Canceled {
		t.Fatalf("persist context error = %v, want context.Canceled after interaction completion", persistCtx.Err())
	}
}
