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
	"fmt"

	"dubbo-admin-ai/runtime"
	conversationstore "dubbo-admin-ai/store"
	memorystore "dubbo-admin-ai/store/memory"
)

// MemoryComponent owns the single conversation Store shared by runtime
// consumers. The legacy HistoryMemory accessors remain for compatibility with
// existing component tests; production Agent and Tool code uses GetStore.
type MemoryComponent struct {
	instanceName string
	historyKey   HistoryKey
	maxTurns     int
	memoryCtx    context.Context
	memory       *HistoryMemory
	store        conversationstore.Store
}

func NewMemoryComponent(historyKey HistoryKey, maxTurns ...int) (runtime.Component, error) {
	limit := 100
	if len(maxTurns) > 0 {
		limit = maxTurns[0]
	}
	return &MemoryComponent{
		historyKey: historyKey,
		maxTurns:   limit,
	}, nil
}

func (m *MemoryComponent) Name() string {
	if m.instanceName != "" {
		return m.instanceName
	}
	return "memory"
}

func (m *MemoryComponent) SetName(name string) {
	m.instanceName = name
}

func (m *MemoryComponent) Validate() error {
	if m.maxTurns <= 0 {
		return fmt.Errorf("max_turns must be greater than 0")
	}
	return nil
}

func (m *MemoryComponent) Init(rt *runtime.Runtime) error {
	m.store = memorystore.NewMemoryStore()
	m.memoryCtx = NewMemoryContext(m.historyKey)
	history, err := GetHistoryMemory(m.memoryCtx, m.historyKey)
	if err != nil {
		return fmt.Errorf("failed to initialize history: %w", err)
	}
	m.memory = history

	rt.GetLogger().Info("Memory component initialized",
		"history_key", m.historyKey)

	return nil
}

func (m *MemoryComponent) Start() error {
	return nil
}

func (m *MemoryComponent) Stop() error {
	return nil
}

// GetContext returns the memory context
func (m *MemoryComponent) GetContext() context.Context {
	return m.memoryCtx
}

// GetStore returns the single conversation Store shared by runtime consumers.
func (m *MemoryComponent) GetStore() (conversationstore.Store, error) {
	if m.store == nil {
		return nil, fmt.Errorf("store not initialized")
	}
	return m.store, nil
}

// GetMemory returns the legacy HistoryMemory instance for compatibility.
func (m *MemoryComponent) GetMemory() (*HistoryMemory, error) {
	if m.memory == nil {
		return nil, fmt.Errorf("history not initialized")
	}
	return m.memory, nil
}
