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

	"dubbo-admin-ai/component/memory"
	"dubbo-admin-ai/component/tools"
	"dubbo-admin-ai/runtime"

	"github.com/firebase/genkit/go/genkit"
)

func TestAgentComponentUsesMemoryComponentStore(t *testing.T) {
	rt := runtime.NewRuntime()
	rt.SetGenkitRegistry(genkit.Init(context.Background()))

	memoryComponentRaw, err := memory.NewMemoryComponent(memory.ChatHistoryKey)
	if err != nil {
		t.Fatalf("NewMemoryComponent() error = %v", err)
	}
	memoryComponent := memoryComponentRaw.(*memory.MemoryComponent)
	if err := memoryComponent.Init(rt); err != nil {
		t.Fatalf("MemoryComponent.Init() error = %v", err)
	}
	rt.RegisterComponent(memoryComponent)
	t.Cleanup(func() { _ = memoryComponent.Stop() })

	toolsComponentRaw, err := tools.NewToolsComponent(tools.ToolConfig{})
	if err != nil {
		t.Fatalf("NewToolsComponent() error = %v", err)
	}
	rt.RegisterComponent(toolsComponentRaw)

	agentComponent := &AgentComponent{}
	if err := agentComponent.Init(rt); err != nil {
		t.Fatalf("AgentComponent.Init() error = %v", err)
	}

	store, err := memoryComponent.GetStore()
	if err != nil {
		t.Fatalf("GetStore() error = %v", err)
	}
	if agentComponent.Agent == nil {
		t.Fatal("AgentComponent.Init() did not create an Agent")
	}
	if agentComponent.Agent.messageStore != store {
		t.Fatalf("agent message store = %p, memory component store = %p; want the same instance", agentComponent.Agent.messageStore, store)
	}
}
