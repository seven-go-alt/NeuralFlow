# Meridian Analytics — Data Breach Incident Response Policy

**Document ID:** doc_compliance_breach
**Owner:** Legal and Compliance
**Last updated:** 2026-02-05

## Purpose

This document defines the steps that should be taken when an employee reports a data breach at Meridian Analytics. A structured incident response process minimizes the impact of security incidents, ensures timely notification to affected parties, and maintains compliance with regulatory obligations under GDPR, CCPA, and other applicable data protection laws.

## Scope

This policy applies to all security incidents involving unauthorized access, disclosure, or loss of Meridian Analytics data, including customer data processed through the Meridian SaaS platform, employee personal data, and internal business data. Every employee, contractor, and third-party vendor is required to report suspected data breaches immediately through the channels described below.

## Steps to take when an employee reports a data breach

When an employee reports a data breach, the following steps must be taken in order:

1. **Containment:** The IT Security team immediately isolates affected systems to prevent further unauthorized access. If the breach involves a compromised account, the account is suspended within 15 minutes of the report. Network segments containing affected resources are placed into quarantine mode.

2. **Triage and assessment:** The incident responder on call assesses the scope of the breach, including what data types were exposed, how many records are affected, and whether the breach is ongoing. A preliminary severity rating is assigned: Low (non-sensitive data, fewer than 100 records), Medium (personal data, 100 to 10,000 records), or Critical (sensitive personal data or more than 10,000 records).

3. **Notification to Legal and Compliance:** The incident responder notifies the Legal and Compliance team within 1 hour of the report. For Critical incidents, the Chief Information Security Officer and Chief Executive Officer are also notified. Legal determines whether regulatory notification obligations are triggered.

4. **Evidence preservation:** The IT Security team captures forensic evidence, including system logs, network traffic captures, and access audit trails. A chain of custody document is created for all collected evidence. No affected systems are restored until forensic analysis is complete.

5. **Regulatory notification:** If required, the Legal team notifies the relevant supervisory authority within 72 hours of becoming aware of the breach. Affected data subjects are notified without undue delay. The notification includes the nature of the breach, the categories of data affected, and recommended mitigation measures.

6. **Remediation and root cause analysis:** The IT Security team conducts a root cause analysis and implements remediation measures to prevent recurrence. The remediation plan is documented and reviewed by the CISO.

7. **Post-incident review:** Within 30 days of containment, the incident response team conducts a post-incident review. Lessons learned are documented, and the incident response plan is updated as needed.

## Employee reporting channels

Employees can report a suspected data breach through any of the following channels:

- **IT Security hotline:** Available 24/7 at +1-555-SEC-ALERT
- **Incident response email:** security-incidents@meridian-analytics.com
- **Slack channel:** #security-incidents (monitored 24/7 by the on-call incident responder)
- **In person:** Report to the IT Security team on the 4th floor of headquarters

Employees who report a data breach in good faith are protected from retaliation under Meridian's Whistleblower Policy.

## Training and drills

All employees complete data breach awareness training annually. The incident response team conducts tabletop exercises quarterly and a full-scale simulated breach exercise annually. The most recent full-scale exercise was conducted in January 2026.

## Revision history

This policy was last updated on 5 February 2026. It will be reviewed semi-annually and after any significant security incident.
