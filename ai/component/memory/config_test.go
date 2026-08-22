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
	"path/filepath"
	"strings"
	"testing"

	"dubbo-admin-ai/runtime"
	gormstore "dubbo-admin-ai/store/gorm"
)

func TestMemorySpecValidate(t *testing.T) {
	tests := []struct {
		name string
		spec MemorySpec
		want string
	}{
		{
			name: "gorm requires database",
			spec: MemorySpec{Backend: "gorm", MaxTurns: 1},
			want: "database is required",
		},
		{
			name: "unknown backend",
			spec: MemorySpec{Backend: "unknown", MaxTurns: 1},
			want: "unsupported memory backend",
		},
		{
			name: "idle cannot exceed open",
			spec: MemorySpec{
				Backend:  "gorm",
				MaxTurns: 1,
				Database: &DatabaseSpec{Driver: "mysql", DSN: "dsn", MaxOpenConns: 2, MaxIdleConns: 3},
			},
			want: "must not exceed",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.spec.Validate(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestMemoryComponent_GormBackendLifecycle(t *testing.T) {
	component, err := NewMemoryComponentFromSpec(MemorySpec{
		Backend:    "gorm",
		HistoryKey: ChatHistoryKey,
		MaxTurns:   1,
		Database: &DatabaseSpec{
			Driver:       "sqlite",
			DSN:          filepath.Join(t.TempDir(), "memory.db"),
			MaxOpenConns: 2,
			MaxIdleConns: 1,
		},
	})
	if err != nil {
		t.Fatalf("NewMemoryComponentFromSpec() error = %v", err)
	}
	if err := component.Init(runtime.NewRuntime()); err != nil {
		t.Fatalf("Init() error = %v", err)
	}
	store, err := component.(*MemoryComponent).GetStore()
	if err != nil {
		t.Fatalf("GetStore() error = %v", err)
	}
	gormBackend, ok := store.(*gormstore.GormStore)
	if !ok {
		t.Fatalf("store type = %T, want *gormstore.GormStore", store)
	}
	sqlDB, err := gormBackend.DB().DB()
	if err != nil {
		t.Fatalf("DB() error = %v", err)
	}
	if got := sqlDB.Stats().MaxOpenConnections; got != 2 {
		t.Fatalf("max open connections = %d, want 2", got)
	}
	if err := component.Stop(); err != nil {
		t.Fatalf("Stop() error = %v", err)
	}
	if err := sqlDB.PingContext(context.Background()); err == nil {
		t.Fatal("PingContext() after Stop() succeeded, want closed database")
	}
}
