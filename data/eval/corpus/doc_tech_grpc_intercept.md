# Meridian Analytics — gRPC Interceptor for Authentication Context in Go Services

**Document ID:** doc_tech_grpc_intercept
**Owner:** Service Architecture Engineering
**Last updated:** 2026-07-18

## Overview

Meridian Analytics' microservices communicate primarily through gRPC, with several services exposing **server-side streaming** RPCs for real-time document processing status updates and trade data feeds. Every RPC call must carry **authentication context** validated from incoming JWT tokens passed via gRPC **metadata**. This document describes the **gRPC interceptor** implemented in **Go** to attach authentication context to every call, including **server-side streaming** RPCs.

## Unary and Server-Side Streaming Interceptor Architecture

The **gRPC interceptor** is implemented as a server-side unary interceptor and a stream interceptor. Both interceptors extract **authentication context** from incoming gRPC **metadata** and validate the bearer token before the request reaches the handler. The **Go** implementation uses the standard `google.golang.org/grpc` interceptor interfaces:

```go
import (
    "context"
    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/metadata"
    "google.golang.org/grpc/status"
)

// AuthInterceptor implements both unary and stream interceptors.
type AuthInterceptor struct {
    jwtValidator JWTValidator
}

// Unary interceptor adds auth context to simple RPC calls.
func (i *AuthInterceptor) Unary() grpc.UnaryServerInterceptor {
    return func(ctx context.Context, req interface{},
        info *grpc.UnaryServerInfo, handler grpc.UnaryHandler,
    ) (interface{}, error) {
        authCtx, err := i.authenticate(ctx)
        if err != nil {
            return nil, err
        }
        return handler(authCtx, req)
    }
}

// Stream interceptor adds auth context to server-side streaming RPC calls.
func (i *AuthInterceptor) Stream() grpc.StreamServerInterceptor {
    return func(srv interface{}, stream grpc.ServerStream,
        info *grpc.StreamServerInfo, handler grpc.StreamHandler,
    ) error {
        authCtx, err := i.authenticate(stream.Context())
        if err != nil {
            return err
        }
        wrappedStream := &authServerStream{ServerStream: stream, ctx: authCtx}
        return handler(srv, wrappedStream)
    }
}
```

## Authentication Context Extraction from Metadata

For every gRPC call, including **server-side streaming** RPCs, the interceptor extracts the bearer token from gRPC **metadata**:

```go
func (i *AuthInterceptor) authenticate(ctx context.Context) (context.Context, error) {
    md, ok := metadata.FromIncomingContext(ctx)
    if !ok {
        return nil, status.Error(codes.Unauthenticated, "missing metadata")
    }

    tokenValues := md["authorization"]
    if len(tokenValues) == 0 {
        return nil, status.Error(codes.Unauthenticated, "missing authorization token")
    }

    token := strings.TrimPrefix(tokenValues[0], "Bearer ")
    claims, err := i.jwtValidator.Validate(token)
    if err != nil {
        return nil, status.Error(codes.Unauthenticated, "invalid token: "+err.Error())
    }

    // Inject validated claims into context for downstream handlers
    authCtx := context.WithValue(ctx, authKey, claims)
    return authCtx, nil
}
```

## Server-Side Streaming Context Propagation

The critical difference for **server-side streaming** RPCs is that the gRPC `ServerStream` interface returns its own context through `stream.Context()`. The interceptor wraps the stream to override the context with the authenticated one. This ensures that every message sent or received on the stream carries the correct **authentication context**:

```go
type authServerStream struct {
    grpc.ServerStream
    ctx context.Context
}

func (s *authServerStream) Context() context.Context {
    return s.ctx
}
```

When the interceptor is applied to a **server-side streaming** RPC, the handler receives a stream whose `Context()` method returns the context enriched with tenant ID, user roles, and token expiry — all derived from the original request **metadata**. This pattern is used by Meridian's document status stream and trade notification stream endpoints.

## Registration and Usage

The **gRPC interceptor** is registered at server initialization time in all **Go** services:

```go
authInterceptor := &AuthInterceptor{jwtValidator: validator}
server := grpc.NewServer(
    grpc.UnaryInterceptor(authInterceptor.Unary()),
    grpc.StreamInterceptor(authInterceptor.Stream()),
)
```

This registration pattern ensures that no RPC — unary or **server-side streaming** — can reach a handler without first passing through the authentication layer. The implementation is shared as an internal library (`go.meridian.io/grpc-auth`) used by all 12 **Go** microservices in Meridian's production fleet.

## Revision History

This document was last updated on 18 July 2026 following the addition of server-side streaming interceptor support for the real-time document status feed.
