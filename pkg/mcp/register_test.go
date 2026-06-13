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

package mcp

import "testing"

func TestRegisterServiceDetailTools(t *testing.T) {
	server := NewServer("test", "dev")
	RegisterTools(server)

	detail, ok := server.tools["get_service_detail"]
	if !ok {
		t.Fatal("Tool 'get_service_detail' not registered")
	}
	if detail.Handler == nil {
		t.Fatal("Tool 'get_service_detail' handler is nil")
	}
	if len(detail.InputSchema.Required) != 1 || detail.InputSchema.Required[0] != "serviceName" {
		t.Fatalf("Expected serviceName to be required, got %v", detail.InputSchema.Required)
	}
	for _, prop := range []string{"serviceName", "version", "group", "mesh"} {
		if _, ok := detail.InputSchema.Properties[prop]; !ok {
			t.Fatalf("get_service_detail missing property %q", prop)
		}
	}
	if _, ok := detail.InputSchema.Properties["side"]; ok {
		t.Fatal("get_service_detail should not expose side")
	}

	distribution, ok := server.tools["get_service_distribution"]
	if !ok {
		t.Fatal("Tool 'get_service_distribution' not registered")
	}
	if distribution.Handler == nil {
		t.Fatal("Tool 'get_service_distribution' handler is nil")
	}
	for _, prop := range []string{"serviceName", "version", "group", "side", "mesh"} {
		if _, ok := distribution.InputSchema.Properties[prop]; !ok {
			t.Fatalf("get_service_distribution missing property %q", prop)
		}
	}
}
