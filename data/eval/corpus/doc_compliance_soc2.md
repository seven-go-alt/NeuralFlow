# Meridian Analytics — SOC 2 Type II Audit Findings Report

**Document ID:** doc_compliance_soc2
**Owner:** Security and Compliance
**Last updated:** 2026-03-10

## Purpose

This document summarizes the SOC 2 Type II audit findings related to access control from the most recent audit report at Meridian Analytics. The SOC 2 Type II audit evaluates the effectiveness of controls over a specified period. The most recent audit covered the period from 1 January 2025 through 31 December 2025 and was conducted by an independent CPA firm.

## Scope of the SOC 2 Type II audit

The SOC 2 Type II audit assessed Meridian's controls across five trust service criteria: security, availability, processing integrity, confidentiality, and privacy. The audit covered the Meridian Analytics SaaS platform, including the data ingestion pipeline, the document processing engine, the embedding generation service, the search and retrieval API, and the customer-facing dashboard. The scope also included internal supporting systems such as the corporate identity provider, the incident management system, and the change management platform.

## SOC 2 Type II audit findings related to access control

The most recent SOC 2 Type II audit findings related to access control identified the following results:

**Control area: User access provisioning and deprovisioning**
- **Finding:** Two test exceptions were identified where deactivated contractor accounts were not removed from the production environment within the required 24-hour SLA. The accounts were disabled in the identity provider but remained active in the database access layer.
- **Remediation:** The Engineering and IT Security teams have implemented an automated reconciliation script that cross-checks active database accounts against the identity provider's disabled user list every 12 hours. This control was implemented in February 2026 and will be tested in the next audit period.

**Control area: Multi-factor authentication**
- **Finding:** No exceptions. Multi-factor authentication was enabled and enforced for all production system access across all user categories throughout the audit period. This control operated effectively.

**Control area: Access reviews**
- **Finding:** Quarterly access reviews were conducted for all 47 production systems. One review in Q2 2025 was completed 5 business days late due to staffing changes. The review ultimately covered all required systems and user accounts.
- **Remediation:** Access review scheduling has been automated with mandatory deadline notifications. A backup reviewer has been assigned for each system.

**Control area: Segregation of duties**
- **Finding:** No exceptions. Segregation of duties controls between development, testing, and production environments were operating effectively. No developer has direct write access to production databases.

**Control area: Remote access and VPN**
- **Finding:** No exceptions. All remote access to production systems required VPN connectivity with device compliance validation. VPN logs showed 100% compliance with the requirement.

## Overall audit opinion

The independent auditor issued an unqualified opinion, indicating that Meridian's controls were suitably designed and operating effectively throughout the audit period. The two exceptions in access control were classified as minor and did not affect the overall audit opinion. The auditor did not identify any material weaknesses or significant deficiencies.

## Remediation tracking

Each finding from the SOC 2 Type II audit findings report is tracked in Meridian's compliance management system with an assigned owner, target remediation date, and evidence of completion. The Security and Compliance team reports remediation status quarterly to the Board of Directors.

## Revision history

This report was last updated on 10 March 2026. The next SOC 2 Type II audit will cover the period from 1 January 2026 through 31 December 2026.
