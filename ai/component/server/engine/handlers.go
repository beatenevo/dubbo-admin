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

package engine

import (
	"errors"
	"fmt"
	"net/http"

	"dubbo-admin-ai/component/agent"
	"dubbo-admin-ai/component/server/engine/session"
	"dubbo-admin-ai/component/server/engine/sse"
	rt "dubbo-admin-ai/runtime"
	"dubbo-admin-ai/schema"
	conversationstore "dubbo-admin-ai/store"

	"github.com/gin-gonic/gin"
)

// AgentHandler handles AI Agent requests.
type AgentHandler struct {
	agent      agent.Agent
	sessionMgr *session.Manager
}

// NewAgentHandler creates an AI Agent handler.
func NewAgentHandler(agent agent.Agent, sessionMgr *session.Manager) *AgentHandler {
	return &AgentHandler{
		agent:      agent,
		sessionMgr: sessionMgr,
	}
}

// StreamChat handles streaming chat endpoint.
func (h *AgentHandler) StreamChat(c *gin.Context) {
	var (
		req          ChatRequest
		pageContext  *schema.AIContextSnapshot
		sessionID    string
		sseHandler   *sse.SSEHandler
		streamWriter *sse.StreamWriter
		channels     *agent.Channels
		err          error
	)

	if err = c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, NewErrorResponse("Invalid request: "+err.Error()))
		return
	}
	if pageContext, err = req.ParseContext(); err != nil {
		c.JSON(http.StatusBadRequest, NewErrorResponse("Invalid request: "+err.Error()))
		return
	}

	sessionID = req.SessionID
	requestCtx := c.Request.Context()
	if _, err = h.sessionMgr.GetSession(requestCtx, sessionID); err != nil {
		h.writeSessionError(c, "Invalid session ID", err, http.StatusBadRequest)
		return
	}
	if err = h.sessionMgr.TouchSession(requestCtx, sessionID); err != nil {
		h.writeSessionError(c, "Invalid session ID", err, http.StatusBadRequest)
		return
	}

	if streamWriter, err = sse.NewStreamWriter(c); err != nil {
		c.JSON(http.StatusInternalServerError, NewErrorResponse("Failed to create stream writer: "+err.Error()))
		return
	}
	sseHandler = sse.NewStreamHandler(streamWriter, sessionID)

	defer func() {
		if r := recover(); r != nil {
			sseHandler.HandleError("internal_error", fmt.Sprintf("internal error: %v", r))
		}
	}()

	channels = h.agent.Interact(requestCtx, &schema.UserInput{Content: req.Message, Context: pageContext}, sessionID)
	userRespChan := channels.UserRespChan
	errorChan := channels.ErrorChan
	var (
		feedback *schema.StreamFeedback
		ok       bool
	)
	for {
		select {
		case err, ok = <-errorChan:
			if !ok {
				errorChan = nil
				continue
			}
			if err != nil {
				sseHandler.HandleError("agent_error", fmt.Sprintf("agent error: %v", err))
				rt.GetLogger().Error("Agent interaction error", "session_id", sessionID, "error", err)
				channels.Close()
				return
			}
		case feedback, ok = <-userRespChan:
			if !ok {
				userRespChan = nil
				continue
			}
			rt.GetLogger().Info("Handler received feedback",
				"session_id", sessionID,
				"text", feedback.Text(),
				"done", feedback.IsDone(),
				"final", feedback.IsFinal(),
				"final_nil", feedback.Final() == nil)
			if feedback.IsFinal() {
				rt.GetLogger().Info("MessageDelta called with output type", "type", fmt.Sprintf("%T", feedback.Final()))
				h.MessageDelta(sseHandler, feedback.Final())
			} else if feedback.IsDone() {
				if err := sseHandler.HandleContentBlockStop(feedback.Index()); err != nil {
					rt.GetLogger().Error("Failed to handle content block stop", "error", err)
				}
			} else {
				if err := sseHandler.HandleText(feedback.Text(), feedback.Index()); err != nil {
					rt.GetLogger().Error("Failed to handle text", "error", err)
				}
			}

		case <-requestCtx.Done():
			rt.GetLogger().Info("Client disconnected from stream")
			return

		case <-channels.Done():
			rt.GetLogger().Info("Channels closed, draining remaining messages", "session_id", sessionID)
		drainLoop:
			for {
				select {
				case feedback, ok = <-userRespChan:
					if !ok {
						userRespChan = nil
						break drainLoop
					}
					h.writeFeedback(sseHandler, feedback)
				case err, ok = <-errorChan:
					if !ok {
						errorChan = nil
						break drainLoop
					}
					if err != nil {
						sseHandler.HandleError("agent_error", fmt.Sprintf("agent error: %v", err))
					}
				default:
					break drainLoop
				}
			}
			if err := sseHandler.FinishStream(); err != nil {
				rt.GetLogger().Error("Failed to finish stream", "error", err)
			}
			rt.GetLogger().Info("Stream processing completed", "session_id", sessionID)
			return
		}
	}
}

func (h *AgentHandler) writeFeedback(sseHandler *sse.SSEHandler, feedback *schema.StreamFeedback) {
	if feedback.IsFinal() {
		h.MessageDelta(sseHandler, feedback.Final())
	} else if feedback.IsDone() {
		if err := sseHandler.HandleContentBlockStop(feedback.Index()); err != nil {
			rt.GetLogger().Error("Failed to handle content block stop", "error", err)
		}
	} else if err := sseHandler.HandleText(feedback.Text(), feedback.Index()); err != nil {
		rt.GetLogger().Error("Failed to handle text", "error", err)
	}
}

// MessageDelta finishes the stream and reports token usage.
func (h *AgentHandler) MessageDelta(sseHandler *sse.SSEHandler, output schema.Schema) {
	stopReason := "end_turn"

	if err := sseHandler.MessageDeltaWithUsage(stopReason, output); err != nil {
		sseHandler.HandleError("finish_stream_error", fmt.Sprintf("failed to finish stream: %v", err))
	}
}

func (h *AgentHandler) writeSessionError(c *gin.Context, message string, err error, invalidStatus int) {
	status := http.StatusInternalServerError
	responseMessage := "Session storage unavailable"
	if errors.Is(err, conversationstore.ErrSessionNotFound) || errors.Is(err, conversationstore.ErrSessionExpired) {
		status = invalidStatus
		responseMessage = message + ": " + err.Error()
	}
	rt.GetLogger().Error("Session store request failed", "error", err)
	c.JSON(status, NewErrorResponse(responseMessage))
}

func (h *AgentHandler) CreateSession(c *gin.Context) {
	sessionObj, err := h.sessionMgr.CreateSession(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusInternalServerError, NewErrorResponse("Failed to create session: "+err.Error()))
		return
	}
	c.JSON(http.StatusOK, NewSuccessResponse(toSessionInfo(sessionObj)))
}

func (h *AgentHandler) GetSession(c *gin.Context) {
	sessionID := c.Param("sessionId")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, NewErrorResponse("Session ID is required"))
		return
	}

	sessionObj, err := h.sessionMgr.GetSession(c.Request.Context(), sessionID)
	if err != nil {
		h.writeSessionError(c, "Session not found", err, http.StatusNotFound)
		return
	}
	c.JSON(http.StatusOK, NewSuccessResponse(toSessionInfo(sessionObj)))
}

func (h *AgentHandler) ListSessions(c *gin.Context) {
	sessionObjs, err := h.sessionMgr.ListSessions(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusInternalServerError, NewErrorResponse("Failed to list sessions: "+err.Error()))
		return
	}
	sessions := make([]map[string]any, 0, len(sessionObjs))
	for _, sessionObj := range sessionObjs {
		sessions = append(sessions, toSessionInfo(sessionObj))
	}

	response := map[string]any{
		"sessions": sessions,
		"total":    len(sessions),
	}
	c.JSON(http.StatusOK, NewSuccessResponse(response))
}

// DeleteSession deletes a session and its conversation history.
func (h *AgentHandler) DeleteSession(c *gin.Context) {
	sessionID := c.Param("sessionId")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, NewErrorResponse("Session ID is required"))
		return
	}

	if err := h.sessionMgr.DeleteSession(c.Request.Context(), sessionID); err != nil {
		h.writeSessionError(c, "Session not found", err, http.StatusNotFound)
		return
	}

	c.JSON(http.StatusOK, NewSuccessResponse(map[string]string{
		"message": "Session deleted successfully",
	}))
}

func toSessionInfo(sessionObj *session.Session) map[string]any {
	return map[string]any{
		"session_id": sessionObj.ID,
		"created_at": sessionObj.CreatedAt,
		"updated_at": sessionObj.UpdatedAt,
		"status":     sessionObj.Status,
	}
}
