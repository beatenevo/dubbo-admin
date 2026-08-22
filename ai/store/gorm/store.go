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

package gormstore

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	conversationstore "dubbo-admin-ai/store"

	"github.com/firebase/genkit/go/ai"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

const sessionExpiration = 24 * time.Hour

// DefaultTurnLimit matches the in-memory store's bounded turn window.
const DefaultTurnLimit = 10

// GormStore persists sessions, turns, and messages in a relational database.
// It deliberately does not use database foreign keys: relationship checks and
// deletion ordering are handled explicitly by the store transaction.
type GormStore struct {
	db    *gorm.DB
	limit int
}

var _ conversationstore.Store = (*GormStore)(nil)

// NewGormStore creates a store around an already opened Gorm database. The
// optional limit exists for tests and preserves the memory backend default.
func NewGormStore(db *gorm.DB, limits ...int) (*GormStore, error) {
	if db == nil {
		return nil, fmt.Errorf("gorm database is nil")
	}
	// The Store owns relationship validation and deletion ordering. Keep Gorm
	// from creating database foreign-key constraints if models gain fields in
	// the future.
	if db.Config == nil {
		db.Config = &gorm.Config{}
	}
	db.Config.DisableForeignKeyConstraintWhenMigrating = true
	limit := DefaultTurnLimit
	if len(limits) > 0 && limits[0] > 0 {
		limit = limits[0]
	}
	return &GormStore{db: db, limit: limit}, nil
}

// Migrate creates or updates the Store tables. Gorm's model associations are
// not declared, so this migration does not create foreign-key constraints.
func (s *GormStore) Migrate(ctx context.Context) error {
	if err := s.checkContext(ctx); err != nil {
		return err
	}
	return s.db.WithContext(normalizeContext(ctx)).AutoMigrate(
		&SessionModel{}, &TurnModel{}, &MessageModel{},
	)
}

// DB returns the underlying database for connection-pool configuration and
// test inspection. Callers must not replace the database instance.
func (s *GormStore) DB() *gorm.DB { return s.db }

// Close closes the underlying SQL database connection.
func (s *GormStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	sqlDB, err := s.db.DB()
	if err != nil {
		return err
	}
	return sqlDB.Close()
}

func (s *GormStore) Create(ctx context.Context, session *conversationstore.Session) error {
	if err := s.checkContext(ctx); err != nil {
		return err
	}
	if err := validateSession(session); err != nil {
		return err
	}
	model := sessionModelFromDomain(session)
	return s.db.WithContext(normalizeContext(ctx)).Create(&model).Error
}

func (s *GormStore) Get(ctx context.Context, sessionID string) (*conversationstore.Session, error) {
	if err := s.checkContext(ctx); err != nil {
		return nil, err
	}
	var model SessionModel
	err := s.db.WithContext(normalizeContext(ctx)).Where("id = ?", sessionID).First(&model).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, conversationstore.ErrSessionNotFound
	}
	if err != nil {
		return nil, err
	}
	if isExpired(model.UpdatedAt, time.Now()) {
		return nil, conversationstore.ErrSessionExpired
	}
	return sessionDomainFromModel(&model), nil
}

func (s *GormStore) List(ctx context.Context) ([]*conversationstore.Session, error) {
	if err := s.checkContext(ctx); err != nil {
		return nil, err
	}
	var models []SessionModel
	cutoff := time.Now().Add(-sessionExpiration)
	err := s.db.WithContext(normalizeContext(ctx)).
		Where("status = ? AND updated_at >= ?", "active", cutoff).
		Find(&models).Error
	if err != nil {
		return nil, err
	}
	result := make([]*conversationstore.Session, 0, len(models))
	for i := range models {
		result = append(result, sessionDomainFromModel(&models[i]))
	}
	return result, nil
}

func (s *GormStore) Touch(ctx context.Context, sessionID string, updatedAt time.Time) error {
	if err := s.checkContext(ctx); err != nil {
		return err
	}
	return s.db.WithContext(normalizeContext(ctx)).Transaction(func(tx *gorm.DB) error {
		model, err := s.findSession(tx, sessionID, true)
		if err != nil {
			return err
		}
		if model.Status != "active" {
			return fmt.Errorf("session %q is not active", sessionID)
		}
		if isExpired(model.UpdatedAt, time.Now()) {
			return conversationstore.ErrSessionExpired
		}
		return tx.Model(&SessionModel{}).Where("id = ?", sessionID).Update("updated_at", updatedAt).Error
	})
}

func (s *GormStore) Delete(ctx context.Context, sessionID string) error {
	if err := s.checkContext(ctx); err != nil {
		return err
	}
	return s.db.WithContext(normalizeContext(ctx)).Transaction(func(tx *gorm.DB) error {
		if _, err := s.findSession(tx, sessionID, true); err != nil {
			return err
		}
		return deleteSessionData(tx, sessionID)
	})
}

func (s *GormStore) DeleteExpired(ctx context.Context, now time.Time) (int, error) {
	if err := s.checkContext(ctx); err != nil {
		return 0, err
	}
	cutoff := now.Add(-sessionExpiration)
	var sessions []SessionModel
	tx := s.db.WithContext(normalizeContext(ctx))
	if err := tx.Where("updated_at < ?", cutoff).Find(&sessions).Error; err != nil {
		return 0, err
	}
	if len(sessions) == 0 {
		return 0, nil
	}
	deleted := 0
	err := tx.Transaction(func(tx *gorm.DB) error {
		for i := range sessions {
			if err := deleteSessionData(tx, sessions[i].ID); err != nil {
				return err
			}
			deleted++
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	return deleted, nil
}

func (s *GormStore) AddHistory(ctx context.Context, sessionID string, messages ...*ai.Message) error {
	if err := s.checkContext(ctx); err != nil {
		return err
	}
	return s.db.WithContext(normalizeContext(ctx)).Transaction(func(tx *gorm.DB) error {
		model, err := s.findSession(tx, sessionID, true)
		if err != nil {
			return err
		}
		if model.Status != "active" {
			return fmt.Errorf("session %q is not active", sessionID)
		}
		if isExpired(model.UpdatedAt, time.Now()) {
			return conversationstore.ErrSessionExpired
		}

		var turn TurnModel
		err = tx.Where("session_id = ? AND completed_at IS NULL", sessionID).
			Order("id DESC").First(&turn).Error
		if errors.Is(err, gorm.ErrRecordNotFound) {
			turn = TurnModel{SessionID: sessionID, CreatedAt: time.Now()}
			if err := tx.Create(&turn).Error; err != nil {
				return err
			}
		} else if err != nil {
			return err
		}

		var last MessageModel
		err = tx.Where("turn_id = ?", turn.ID).Order("sequence DESC").First(&last).Error
		nextSequence := uint64(0)
		if err == nil {
			nextSequence = last.Sequence + 1
		} else if !errors.Is(err, gorm.ErrRecordNotFound) {
			return err
		}
		for _, message := range messages {
			if message == nil || !supportedRole(message.Role) {
				continue
			}
			payload, err := encodeMessage(message)
			if err != nil {
				return fmt.Errorf("failed to encode message: %w", err)
			}
			stored := MessageModel{
				TurnID:    turn.ID,
				Sequence:  nextSequence,
				Payload:   payload,
				CreatedAt: time.Now(),
			}
			if err := tx.Create(&stored).Error; err != nil {
				return err
			}
			nextSequence++
		}
		return nil
	})
}

func (s *GormStore) IsEmpty(ctx context.Context, sessionID string) (bool, error) {
	if err := s.checkContext(ctx); err != nil {
		return false, err
	}
	var turn TurnModel
	err := s.db.WithContext(normalizeContext(ctx)).
		Where("session_id = ? AND completed_at IS NULL", sessionID).
		First(&turn).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return true, nil
	}
	return false, err
}

func (s *GormStore) WindowMemory(ctx context.Context, sessionID string) ([]*ai.Message, error) {
	if err := s.checkContext(ctx); err != nil {
		return nil, err
	}
	var turn TurnModel
	err := s.db.WithContext(normalizeContext(ctx)).
		Where("session_id = ? AND completed_at IS NULL", sessionID).
		First(&turn).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return s.readTurnMessages(ctx, turn.ID)
}

func (s *GormStore) AllMemory(ctx context.Context, sessionID string) ([]*ai.Message, error) {
	if err := s.checkContext(ctx); err != nil {
		return nil, err
	}
	var active *TurnModel
	var activeModel TurnModel
	err := s.db.WithContext(normalizeContext(ctx)).
		Where("session_id = ? AND completed_at IS NULL", sessionID).
		First(&activeModel).Error
	if err == nil {
		active = &activeModel
	} else if !errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, err
	}

	var completed []TurnModel
	if err := s.db.WithContext(normalizeContext(ctx)).
		Where("session_id = ? AND completed_at IS NOT NULL", sessionID).
		Order("id ASC").Find(&completed).Error; err != nil {
		return nil, err
	}
	turns := make([]TurnModel, 0, len(completed)+1)
	if active != nil {
		turns = append(turns, *active)
	}
	turns = append(turns, completed...)
	var result []*ai.Message
	for i := range turns {
		messages, err := s.readTurnMessages(ctx, turns[i].ID)
		if err != nil {
			return nil, err
		}
		result = append(result, messages...)
	}
	return result, nil
}

func (s *GormStore) NextTurn(ctx context.Context, sessionID string) error {
	if err := s.checkContext(ctx); err != nil {
		return err
	}
	return s.db.WithContext(normalizeContext(ctx)).Transaction(func(tx *gorm.DB) error {
		model, err := s.findSession(tx, sessionID, true)
		if err != nil {
			return err
		}
		if model.Status != "active" {
			return fmt.Errorf("session %q is not active", sessionID)
		}
		if isExpired(model.UpdatedAt, time.Now()) {
			return conversationstore.ErrSessionExpired
		}
		var active TurnModel
		if err := tx.Where("session_id = ? AND completed_at IS NULL", sessionID).First(&active).Error; errors.Is(err, gorm.ErrRecordNotFound) {
			return conversationstore.ErrNoActiveTurn
		} else if err != nil {
			return err
		}
		var count int64
		if err := tx.Model(&TurnModel{}).Where("session_id = ?", sessionID).Count(&count).Error; err != nil {
			return err
		}
		if count >= int64(s.limit) {
			return fmt.Errorf("%w: current session's context is full, please create a new session", conversationstore.ErrTurnLimitReached)
		}
		completedAt := time.Now()
		return tx.Model(&TurnModel{}).Where("id = ?", active.ID).Update("completed_at", completedAt).Error
	})
}

func (s *GormStore) readTurnMessages(ctx context.Context, turnID uint64) ([]*ai.Message, error) {
	var models []MessageModel
	if err := s.db.WithContext(normalizeContext(ctx)).Where("turn_id = ?", turnID).Order("sequence ASC").Find(&models).Error; err != nil {
		return nil, err
	}
	return decodeGrouped(models)
}

func decodeGrouped(models []MessageModel) ([]*ai.Message, error) {
	system := make([]*ai.Message, 0, len(models))
	user := make([]*ai.Message, 0, len(models))
	model := make([]*ai.Message, 0, len(models))
	for i := range models {
		message, err := decodeMessage(models[i].ID, models[i].Payload)
		if err != nil {
			return nil, err
		}
		switch message.Role {
		case ai.RoleSystem:
			system = append(system, message)
		case ai.RoleUser:
			user = append(user, message)
		case ai.RoleModel:
			model = append(model, message)
		}
	}
	result := make([]*ai.Message, 0, len(system)+len(user)+len(model))
	result = append(result, system...)
	result = append(result, user...)
	result = append(result, model...)
	return result, nil
}

func (s *GormStore) findSession(tx *gorm.DB, sessionID string, lock bool) (*SessionModel, error) {
	query := tx.Where("id = ?", sessionID)
	if lock && supportsRowLock(tx) {
		query = query.Clauses(clause.Locking{Strength: "UPDATE"})
	}
	var model SessionModel
	if err := query.First(&model).Error; errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, conversationstore.ErrSessionNotFound
	} else if err != nil {
		return nil, err
	}
	return &model, nil
}

func deleteSessionData(tx *gorm.DB, sessionID string) error {
	var turns []TurnModel
	if err := tx.Where("session_id = ?", sessionID).Find(&turns).Error; err != nil {
		return err
	}
	if len(turns) > 0 {
		ids := make([]uint64, 0, len(turns))
		for i := range turns {
			ids = append(ids, turns[i].ID)
		}
		if err := tx.Where("turn_id IN ?", ids).Delete(&MessageModel{}).Error; err != nil {
			return err
		}
	}
	if err := tx.Where("session_id = ?", sessionID).Delete(&TurnModel{}).Error; err != nil {
		return err
	}
	return tx.Where("id = ?", sessionID).Delete(&SessionModel{}).Error
}

func validateSession(session *conversationstore.Session) error {
	if session == nil {
		return errors.New("session is nil")
	}
	if strings.TrimSpace(session.ID) == "" {
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
	return nil
}

func sessionModelFromDomain(session *conversationstore.Session) SessionModel {
	return SessionModel{ID: session.ID, CreatedAt: session.CreatedAt, UpdatedAt: session.UpdatedAt, Status: session.Status}
}

func sessionDomainFromModel(model *SessionModel) *conversationstore.Session {
	return &conversationstore.Session{ID: model.ID, CreatedAt: model.CreatedAt, UpdatedAt: model.UpdatedAt, Status: model.Status}
}

func (s *GormStore) checkContext(ctx context.Context) error {
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

func normalizeContext(ctx context.Context) context.Context {
	if ctx == nil {
		return context.Background()
	}
	return ctx
}

func supportsRowLock(tx *gorm.DB) bool {
	name := strings.ToLower(tx.Name())
	return name == "mysql" || name == "postgres"
}

func supportedRole(role ai.Role) bool {
	return role == ai.RoleSystem || role == ai.RoleUser || role == ai.RoleModel
}

func isExpired(updatedAt, now time.Time) bool {
	return now.Sub(updatedAt) > sessionExpiration
}
