# Meridian Analytics — Production Deployment Governance and Checklist

**Document ID:** doc_tech_deploy_checklist
**Owner:** Platform Engineering
**Last updated:** 2026-07-08

## Purpose

This document defines the mandatory **pre-deployment checklist** required before any **production deployment** at Meridian Analytics. Skip or bypass this **pre-deployment checklist** at your own risk — the consequences of doing so are documented below.

## Pre-Deployment Checklist

Every **production deployment** must complete the following **pre-deployment checklist** items, verified in the Meridian deployment portal:

| Item | Check | Verified by |
|------|-------|-------------|
| All unit and integration tests pass | ☐ | CI pipeline |
| Security scan (Trivy) reports no critical or high vulnerabilities | ☐ | Security Engineering |
| Database migration script reviewed and approved | ☐ | Database Engineering |
| Rollback plan documented and tested | ☐ | Deploying engineer |
| Feature flags configured for gradual rollout | ☐ | Product Engineering |
| Performance benchmark results within SLA thresholds | ☐ | Platform Engineering |
| No active **deployment freeze** window | ☐ | Release manager |
| Change advisory board (CAB) approval for significant changes | ☐ | Engineering leadership |

## Consequences of Skipping the Pre-Deployment Checklist

If a **production deployment** is attempted without completing the **pre-deployment checklist**, the following consequences can occur:

1. **Rollback delays:** Without a documented and tested rollback plan, reverting a faulty deployment takes 3–5x longer. Meridian's SRE team recorded an average rollback time of 47 minutes for checklist-compliant deployments versus 192 minutes for non-compliant ones in Q2 2026.

2. **Silent data corruption:** Skipping the database migration review on the **pre-deployment checklist** can result in schema changes that silently corrupt existing records. One such incident in March 2026 caused 12 hours of data inconsistency across 40,000 customer documents.

3. **Security exposure:** Deploying without passing the Trivy security scan has led to unpatched CVEs in production. A skipped security scan in the **pre-deployment checklist** was the root cause of a critical vulnerability (CVE-2026-1234) being deployed to the search API fleet in February 2026.

4. **Deployment freeze violation:** Pushing a **production deployment** during a **deployment freeze** window (e.g., end-of-quarter, major product launch, or holiday period) without an approved exception violates Meridian's change management policy and may result in disciplinary action up to and including suspension of deployment privileges.

## Deployment Freeze Policy

A **deployment freeze** is enforced during the following periods:

- Last 7 calendar days of each quarter (freeze window for financial reporting stability).
- 48 hours before and 24 hours after any major product launch.
- The entire month of December (holiday freeze).
- Any period declared by the CTO due to ongoing incident or instability.

During a **deployment freeze**, only emergency patches approved by the VP of Engineering and the Security lead are permitted, and only if the **pre-deployment checklist** has been reviewed and signed off.

## Revision History

This document was last updated on 8 July 2026 to incorporate findings from the Q2 incident postmortems.
