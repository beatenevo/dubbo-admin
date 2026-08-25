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

package storetest

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	conversationstore "dubbo-admin-ai/store"

	"github.com/firebase/genkit/go/ai"
)

// RunStoreContractTests verifies behavior shared by every Store backend.
// Backend-specific tests should call this function with their own constructor.
func RunStoreContractTests(t *testing.T, newStore func() conversationstore.Store, limit int) {
	t.Helper()
	if limit <= 0 {
		t.Fatalf("contract test limit must be positive, got %d", limit)
	}

	t.Run("session lifecycle and history deletion", func(t *testing.T) {
		ctx := context.Background()
		s := newStore()
		now := time.Now()
		session := &conversationstore.Session{
			ID:        "contract-session",
			CreatedAt: now,
			UpdatedAt: now,
			Status:    "active",
		}
		if err := s.Create(ctx, session); err != nil {
			t.Fatalf("Create() error = %v", err)
		}
		if err := s.AddHistory(ctx, session.ID, ai.NewUserMessage(ai.NewTextPart("hello"))); err != nil {
			t.Fatalf("AddHistory() error = %v", err)
		}
		if err := s.Delete(ctx, session.ID); err != nil {
			t.Fatalf("Delete() error = %v", err)
		}
		if _, err := s.Get(ctx, session.ID); !errors.Is(err, conversationstore.ErrSessionNotFound) {
			t.Fatalf("Get() after Delete() error = %v, want ErrSessionNotFound", err)
		}
		messages, err := s.AllMemory(ctx, session.ID)
		if err != nil {
			t.Fatalf("AllMemory() after Delete() error = %v", err)
		}
		if len(messages) != 0 {
			t.Fatalf("AllMemory() after Delete() = %d messages, want 0", len(messages))
		}
	})

	t.Run("message roles and turn order", func(t *testing.T) {
		ctx := context.Background()
		s := newStore()
		now := time.Now()
		session := &conversationstore.Session{ID: "order-session", CreatedAt: now, UpdatedAt: now, Status: "active"}
		if err := s.Create(ctx, session); err != nil {
			t.Fatalf("Create() error = %v", err)
		}
		if err := s.AddHistory(ctx, session.ID,
			ai.NewUserMessage(ai.NewTextPart("first")),
			ai.NewSystemMessage(ai.NewTextPart("system")),
			ai.NewMessage(ai.RoleModel, nil, ai.NewTextPart("model")),
			nil,
			ai.NewMessage(ai.RoleTool, nil, ai.NewTextPart("ignored")),
		); err != nil {
			t.Fatalf("AddHistory() error = %v", err)
		}
		if err := s.NextTurn(ctx, session.ID); err != nil {
			t.Fatalf("NextTurn() error = %v", err)
		}
		if err := s.AddHistory(ctx, session.ID, ai.NewUserMessage(ai.NewTextPart("second"))); err != nil {
			t.Fatalf("second AddHistory() error = %v", err)
		}
		messages, err := s.AllMemory(ctx, session.ID)
		if err != nil {
			t.Fatalf("AllMemory() error = %v", err)
		}
		if len(messages) != 4 || messages[0].Content[0].Text != "second" || messages[1].Role != ai.RoleSystem || messages[2].Role != ai.RoleUser || messages[3].Role != ai.RoleModel {
			t.Fatalf("AllMemory() = %#v, want active turn followed by system/user/model", messages)
		}
	})

	t.Run("returned messages are snapshots", func(t *testing.T) {
		ctx := context.Background()
		s := newStore()
		now := time.Now()
		session := &conversationstore.Session{ID: "snapshot-session", CreatedAt: now, UpdatedAt: now, Status: "active"}
		if err := s.Create(ctx, session); err != nil {
			t.Fatalf("Create() error = %v", err)
		}
		message := ai.NewUserMessage(ai.NewTextPart("original"))
		if err := s.AddHistory(ctx, session.ID, message); err != nil {
			t.Fatalf("AddHistory() error = %v", err)
		}
		message.Content[0].Text = "caller mutation"
		stored, err := s.WindowMemory(ctx, session.ID)
		if err != nil {
			t.Fatalf("WindowMemory() error = %v", err)
		}
		if len(stored) != 1 || stored[0].Content[0].Text != "original" {
			t.Fatalf("stored message = %#v, want original snapshot", stored)
		}
	})

	t.Run("empty and unsupported history still creates an active turn", func(t *testing.T) {
		ctx := context.Background()
		s := newStore()
		now := time.Now()
		session := &conversationstore.Session{ID: "empty-turn-session", CreatedAt: now, UpdatedAt: now, Status: "active"}
		if err := s.Create(ctx, session); err != nil {
			t.Fatalf("Create() error = %v", err)
		}
		if err := s.AddHistory(ctx, session.ID, nil, ai.NewMessage(ai.RoleTool, nil, ai.NewTextPart("ignored"))); err != nil {
			t.Fatalf("AddHistory() error = %v", err)
		}
		empty, err := s.IsEmpty(ctx, session.ID)
		if err != nil {
			t.Fatalf("IsEmpty() error = %v", err)
		}
		if empty {
			t.Fatal("IsEmpty() = true after an empty AddHistory(), want false")
		}
		messages, err := s.WindowMemory(ctx, session.ID)
		if err != nil {
			t.Fatalf("WindowMemory() error = %v", err)
		}
		if len(messages) != 0 {
			t.Fatalf("WindowMemory() = %#v, want empty", messages)
		}
		if err := s.NextTurn(ctx, session.ID); err != nil {
			t.Fatalf("NextTurn() on empty active turn error = %v", err)
		}
		if err := s.NextTurn(ctx, session.ID); !errors.Is(err, conversationstore.ErrNoActiveTurn) {
			t.Fatalf("NextTurn() without active turn = %v, want ErrNoActiveTurn", err)
		}
	})

	t.Run("closed session rejects history writes", func(t *testing.T) {
		ctx := context.Background()
		s := newStore()
		now := time.Now()
		session := &conversationstore.Session{ID: "closed-session", CreatedAt: now, UpdatedAt: now, Status: "closed"}
		if err := s.Create(ctx, session); err != nil {
			t.Fatalf("Create() error = %v", err)
		}
		if err := s.AddHistory(ctx, session.ID, ai.NewUserTextMessage("rejected")); err == nil || !strings.Contains(err.Error(), "not active") {
			t.Fatalf("AddHistory() error = %v, want inactive-session error", err)
		}
		if err := s.NextTurn(ctx, session.ID); err == nil || !strings.Contains(err.Error(), "not active") {
			t.Fatalf("NextTurn() error = %v, want inactive-session error", err)
		}
	})

	t.Run("expiration is consistent across session operations", func(t *testing.T) {
		ctx := context.Background()
		s := newStore()
		now := time.Now()
		expired := &conversationstore.Session{
			ID:        "expired-contract-session",
			CreatedAt: now.Add(-25 * time.Hour),
			UpdatedAt: now.Add(-25 * time.Hour),
			Status:    "active",
		}
		active := &conversationstore.Session{ID: "active-contract-session", CreatedAt: now, UpdatedAt: now, Status: "active"}
		for _, session := range []*conversationstore.Session{expired, active} {
			if err := s.Create(ctx, session); err != nil {
				t.Fatalf("Create(%s) error = %v", session.ID, err)
			}
		}
		if _, err := s.Get(ctx, expired.ID); !errors.Is(err, conversationstore.ErrSessionExpired) {
			t.Fatalf("Get(expired) error = %v, want ErrSessionExpired", err)
		}
		if err := s.Touch(ctx, expired.ID, now); !errors.Is(err, conversationstore.ErrSessionExpired) {
			t.Fatalf("Touch(expired) error = %v, want ErrSessionExpired", err)
		}
		listed, err := s.List(ctx)
		if err != nil {
			t.Fatalf("List() error = %v", err)
		}
		if len(listed) != 1 || listed[0].ID != active.ID {
			t.Fatalf("List() = %#v, want only active session", listed)
		}
		deleted, err := s.DeleteExpired(ctx, now)
		if err != nil {
			t.Fatalf("DeleteExpired() error = %v", err)
		}
		if deleted != 1 {
			t.Fatalf("DeleteExpired() = %d, want 1", deleted)
		}
		if _, err := s.Get(ctx, expired.ID); !errors.Is(err, conversationstore.ErrSessionNotFound) {
			t.Fatalf("Get(expired) after DeleteExpired error = %v, want ErrSessionNotFound", err)
		}
	})

	t.Run("turn limit and failed transition preserve active turn", func(t *testing.T) {
		ctx := context.Background()
		s := newStore()
		now := time.Now()
		session := &conversationstore.Session{ID: "limit-session", CreatedAt: now, UpdatedAt: now, Status: "active"}
		if err := s.Create(ctx, session); err != nil {
			t.Fatalf("Create() error = %v", err)
		}
		if err := s.AddHistory(ctx, session.ID, ai.NewUserTextMessage("turn-0")); err != nil {
			t.Fatalf("initial AddHistory() error = %v", err)
		}
		for i := 1; i < limit; i++ {
			if err := s.NextTurn(ctx, session.ID); err != nil {
				t.Fatalf("NextTurn() at turn %d error = %v", i-1, err)
			}
			if err := s.AddHistory(ctx, session.ID, ai.NewUserTextMessage(fmt.Sprintf("turn-%d", i))); err != nil {
				t.Fatalf("AddHistory() at turn %d error = %v", i, err)
			}
		}
		if err := s.NextTurn(ctx, session.ID); err != nil {
			t.Fatalf("final NextTurn() error = %v, want success", err)
		}
		if err := s.AddHistory(ctx, session.ID, ai.NewUserTextMessage("overflow")); !errors.Is(err, conversationstore.ErrTurnLimitReached) {
			t.Fatalf("overflow AddHistory() error = %v, want ErrTurnLimitReached", err)
		}
		messages, err := s.AllMemory(ctx, session.ID)
		if err != nil {
			t.Fatalf("AllMemory() after rejected turn error = %v", err)
		}
		if len(messages) != limit {
			t.Fatalf("AllMemory() after rejected turn = %d messages, want %d", len(messages), limit)
		}
	})
}
