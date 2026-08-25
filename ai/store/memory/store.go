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

package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	conversationstore "dubbo-admin-ai/store"

	"github.com/firebase/genkit/go/ai"
)

// DefaultMaxTurns matches MemorySpec's default conversation limit.
const DefaultMaxTurns = 100

const sessionExpiration = 24 * time.Hour

type turn struct {
	id             uint64
	createdAt      time.Time
	userMessages   []*ai.Message
	modelMessages  []*ai.Message
	systemMessages []*ai.Message
}

func (t *turn) messages() []*ai.Message {
	messages := make([]*ai.Message, 0,
		len(t.systemMessages)+len(t.userMessages)+len(t.modelMessages))
	messages = append(messages, t.systemMessages...)
	messages = append(messages, t.userMessages...)
	messages = append(messages, t.modelMessages...)
	return messages
}

// turnWindow mirrors the current HistoryMemory window's bounded behavior. The
// end index is intentionally monotonic, so a full window remains full after
// repeated Pop/Push cycles just as it does in the existing implementation.
type turnWindow struct {
	limit int
	begin int
	end   int
	data  []*turn
}

func newTurnWindow(limit int) *turnWindow {
	return &turnWindow{limit: limit, data: make([]*turn, limit+1)}
}

func (w *turnWindow) empty() bool {
	return w.begin == w.end
}

func (w *turnWindow) full() bool {
	return w.end == w.limit
}

func (w *turnWindow) push(value *turn) bool {
	if w.full() {
		return false
	}
	w.data[w.end] = value
	w.end++
	return true
}

func (w *turnWindow) pop() *turn {
	if w.empty() {
		return nil
	}
	value := w.data[w.begin]
	w.data[w.begin] = nil
	w.begin++
	return value
}

func (w *turnWindow) values() []*turn {
	return w.data[w.begin:w.end]
}

type sessionHistory struct {
	window  *turnWindow
	history []*turn
	nextID  uint64
}

// MemoryStore is the in-process implementation of the conversation Store.
type MemoryStore struct {
	mu       sync.RWMutex
	limit    int
	sessions map[string]conversationstore.Session
	history  map[string]*sessionHistory
}

var _ conversationstore.Store = (*MemoryStore)(nil)

// NewMemoryStore creates a MemoryStore. The optional limit exists for runtime
// configuration and tests; the default matches MemorySpec.
func NewMemoryStore(limits ...int) *MemoryStore {
	limit := DefaultMaxTurns
	if len(limits) > 0 && limits[0] > 0 {
		limit = limits[0]
	}
	return &MemoryStore{
		limit:    limit,
		sessions: make(map[string]conversationstore.Session),
		history:  make(map[string]*sessionHistory),
	}
}

func (m *MemoryStore) Create(ctx context.Context, session *conversationstore.Session) error {
	if err := checkContext(ctx); err != nil {
		return err
	}
	if session == nil {
		return errors.New("session is nil")
	}
	if session.ID == "" {
		return errors.New("session id is required")
	}
	if session.CreatedAt.IsZero() {
		return errors.New("session created_at is required")
	}
	if session.UpdatedAt.IsZero() {
		return errors.New("session updated_at is required")
	}
	if session.Status == "" {
		return errors.New("session status is required")
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.sessions[session.ID]; exists {
		return fmt.Errorf("session %q already exists", session.ID)
	}
	m.sessions[session.ID] = *session
	return nil
}

func (m *MemoryStore) Get(ctx context.Context, sessionID string) (*conversationstore.Session, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()
	session, exists := m.sessions[sessionID]
	if !exists {
		return nil, conversationstore.ErrSessionNotFound
	}
	if isExpired(session.UpdatedAt, time.Now()) {
		return nil, conversationstore.ErrSessionExpired
	}
	copy := session
	return &copy, nil
}

func (m *MemoryStore) List(ctx context.Context) ([]*conversationstore.Session, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]*conversationstore.Session, 0, len(m.sessions))
	now := time.Now()
	for _, session := range m.sessions {
		if session.Status != "active" || isExpired(session.UpdatedAt, now) {
			continue
		}
		copy := session
		result = append(result, &copy)
	}
	return result, nil
}

func (m *MemoryStore) Touch(ctx context.Context, sessionID string, updatedAt time.Time) error {
	if err := checkContext(ctx); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	session, exists := m.sessions[sessionID]
	if !exists {
		return conversationstore.ErrSessionNotFound
	}
	if session.Status != "active" {
		return fmt.Errorf("session %q is not active", sessionID)
	}
	if isExpired(session.UpdatedAt, time.Now()) {
		return conversationstore.ErrSessionExpired
	}
	session.UpdatedAt = updatedAt
	m.sessions[sessionID] = session
	return nil
}

func (m *MemoryStore) Delete(ctx context.Context, sessionID string) error {
	if err := checkContext(ctx); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.sessions[sessionID]; !exists {
		return conversationstore.ErrSessionNotFound
	}
	delete(m.sessions, sessionID)
	delete(m.history, sessionID)
	return nil
}

func (m *MemoryStore) DeleteExpired(ctx context.Context, now time.Time) (int, error) {
	if err := checkContext(ctx); err != nil {
		return 0, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	deleted := 0
	for sessionID, session := range m.sessions {
		if !isExpired(session.UpdatedAt, now) {
			continue
		}
		delete(m.sessions, sessionID)
		delete(m.history, sessionID)
		deleted++
	}
	return deleted, nil
}

func (m *MemoryStore) AddHistory(ctx context.Context, sessionID string, messages ...*ai.Message) error {
	if err := checkContext(ctx); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.validateSessionLocked(sessionID); err != nil {
		return err
	}

	history := m.ensureHistoryLocked(sessionID)
	if history.window.empty() {
		if len(history.history) >= m.limit {
			return fmt.Errorf("%w: current session's context is full, please create a new session", conversationstore.ErrTurnLimitReached)
		}
		history.nextID++
		if !history.window.push(&turn{id: history.nextID, createdAt: time.Now()}) {
			return fmt.Errorf("failed to create active turn: %w", conversationstore.ErrTurnLimitReached)
		}
	}
	active := history.window.values()[len(history.window.values())-1]
	for _, message := range messages {
		if message == nil || !supportedRole(message.Role) {
			continue
		}
		copy, err := cloneMessage(message)
		if err != nil {
			return fmt.Errorf("failed to copy message: %w", err)
		}
		switch copy.Role {
		case ai.RoleSystem:
			active.systemMessages = append(active.systemMessages, copy)
		case ai.RoleUser:
			active.userMessages = append(active.userMessages, copy)
		case ai.RoleModel:
			active.modelMessages = append(active.modelMessages, copy)
		}
	}
	return nil
}

func (m *MemoryStore) IsEmpty(ctx context.Context, sessionID string) (bool, error) {
	if err := checkContext(ctx); err != nil {
		return false, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()
	history := m.history[sessionID]
	return history == nil || history.window == nil || history.window.empty(), nil
}

func (m *MemoryStore) WindowMemory(ctx context.Context, sessionID string) ([]*ai.Message, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()
	history := m.history[sessionID]
	if history == nil || history.window == nil {
		return nil, nil
	}
	return cloneTurnList(history.window.values())
}

func (m *MemoryStore) AllMemory(ctx context.Context, sessionID string) ([]*ai.Message, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()
	history := m.history[sessionID]
	if history == nil || history.window == nil {
		return nil, nil
	}
	turns := make([]*turn, 0, len(history.window.values())+len(history.history))
	turns = append(turns, history.window.values()...)
	turns = append(turns, history.history...)
	return cloneTurnList(turns)
}

func (m *MemoryStore) NextTurn(ctx context.Context, sessionID string) error {
	if err := checkContext(ctx); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if err := m.validateSessionLocked(sessionID); err != nil {
		return err
	}
	history := m.history[sessionID]
	if history == nil || history.window == nil || history.window.empty() {
		return conversationstore.ErrNoActiveTurn
	}
	if len(history.history) >= m.limit {
		return fmt.Errorf("%w: current session's context is full, please create a new session", conversationstore.ErrTurnLimitReached)
	}
	completed := history.window.pop()
	history.history = append(history.history, completed)
	return nil
}

func (m *MemoryStore) validateSessionLocked(sessionID string) error {
	session, exists := m.sessions[sessionID]
	if !exists {
		return conversationstore.ErrSessionNotFound
	}
	if session.Status != "active" {
		return fmt.Errorf("session %q is not active", sessionID)
	}
	if isExpired(session.UpdatedAt, time.Now()) {
		return conversationstore.ErrSessionExpired
	}
	return nil
}

func (m *MemoryStore) ensureHistoryLocked(sessionID string) *sessionHistory {
	history := m.history[sessionID]
	if history == nil {
		history = &sessionHistory{window: newTurnWindow(m.limit)}
		m.history[sessionID] = history
	}
	return history
}

func cloneTurnList(turns []*turn) ([]*ai.Message, error) {
	var result []*ai.Message
	for _, current := range turns {
		if current == nil {
			continue
		}
		for _, message := range current.messages() {
			copy, err := cloneMessage(message)
			if err != nil {
				return nil, fmt.Errorf("failed to copy stored message: %w", err)
			}
			result = append(result, copy)
		}
	}
	return result, nil
}

func cloneMessage(message *ai.Message) (*ai.Message, error) {
	if message == nil {
		return nil, errors.New("message is nil")
	}
	data, err := json.Marshal(message)
	if err != nil {
		return nil, err
	}
	var copy ai.Message
	if err := json.Unmarshal(data, &copy); err != nil {
		return nil, err
	}
	return &copy, nil
}

func supportedRole(role ai.Role) bool {
	return role == ai.RoleSystem || role == ai.RoleUser || role == ai.RoleModel
}

func isExpired(updatedAt, now time.Time) bool {
	return now.Sub(updatedAt) > sessionExpiration
}

func checkContext(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil
	}
}
