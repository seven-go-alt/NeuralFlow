# Meridian Analytics — TLS 1.3 Termination on Kubernetes Ingress

**Document ID:** doc_tech_tls
**Owner:** Infrastructure Engineering
**Last updated:** 2026-07-11

## Overview

Meridian Analytics terminates **TLS 1.3** at the **ingress gateway** for all external HTTP traffic entering the **Kubernetes** cluster. This document describes the configuration for a staging environment using **cert-manager** with **Let's Encrypt** as the certificate issuer.

## Ingress Gateway Configuration

The **ingress gateway** in Meridian's **Kubernetes** cluster is implemented using the NGINX Ingress Controller configured to accept only **TLS 1.3** connections. The minimum TLS version enforcement is critical for Meridian's security compliance requirements:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-ingress-controller
  namespace: ingress-nginx
data:
  ssl-protocols: "TLSv1.3"
  ssl-ciphers: "TLS_AES_128_GCM_SHA256:TLS_AES_256_GCM_SHA384"
  ssl-prefer-server-ciphers: "true"
  use-forwarded-headers: "true"
```

The `ssl-protocols` directive restricts the **ingress gateway** to **TLS 1.3** only, rejecting all handshake attempts using TLS 1.2 or earlier. The cipher suite list selects only the AEAD ciphers defined in RFC 8446.

## Certificate Management with cert-manager

Meridian uses **cert-manager** in the staging **Kubernetes** cluster to automate certificate provisioning from **Let's Encrypt**:

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-staging
spec:
  acme:
    server: https://acme-staging-v02.api.letsencrypt.org/directory
    email: infrastructure@meridiananalytics.com
    privateKeySecretRef:
      name: letsencrypt-staging-key
    solvers:
    - http01:
        ingress:
          class: nginx
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: api-staging-meridian
  namespace: staging
spec:
  secretName: api-staging-tls
  issuerRef:
    name: letsencrypt-staging
    kind: ClusterIssuer
  commonName: api.staging.meridiananalytics.com
  dnsNames:
  - api.staging.meridiananalytics.com
  - ws.staging.meridiananalytics.com
```

**cert-manager** automatically creates the ACME HTTP-01 challenge tokens in the **ingress gateway** configuration and handles the full certificate lifecycle, including renewal before expiration.

## TLS 1.3 Termination at the Ingress

The **TLS 1.3** termination point is the **ingress gateway** running NGINX. The handler decrypts incoming traffic and forwards plain HTTP to backend services within the **Kubernetes** cluster:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-staging
  namespace: staging
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-staging
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.staging.meridiananalytics.com
    secretName: api-staging-tls
  rules:
  - host: api.staging.meridiananalytics.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway
            port:
              number: 8080
```

All HTTP traffic is redirected to HTTPS via `ssl-redirect: "true"`, and the **cert-manager** annotation links the ingress to the ClusterIssuer.

## Staging Environment Considerations

In the staging **Kubernetes** environment, Meridian uses the **Let's Encrypt** staging ACME endpoint to avoid hitting production issuance rate limits during development:

| Feature | Staging | Production |
|---------|---------|------------|
| ACME endpoint | `acme-staging-v02.api.letsencrypt.org` | `acme-v02.api.letsencrypt.org` |
| Certificate chain | Fake CA (untrusted) | Trusted by browsers |
| Rate limit | 30,000 certificates/week | 50 certificates/week |
| Renewal window | 30 days | 30 days |

The staging **cert-manager** issuer uses the same DNS-01 or HTTP-01 solver configuration as production, so validation logic is tested before promoting to production.

## Verification

After deploying the **TLS 1.3** configuration, Meridian verifies termination using:

```bash
openssl s_client -connect api.staging.meridiananalytics.com:443 -tls1_3
```

This confirms that the **ingress gateway** negotiates **TLS 1.3** and that the **cert-manager**-issued certificate from **Let's Encrypt** is presented correctly.

## Revision History

This guide was last updated on 11 July 2026 following the upgrade from TLS 1.2 to **TLS 1.3** across all staging ingresses.
