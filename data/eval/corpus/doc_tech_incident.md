# Meridian Analytics — Critical Incident Escalation Runbook

**Document ID:** doc_tech_incident
**Owner:** Site Reliability Engineering
**Last updated:** 2026-06-28

## Incident Classification

Meridian Analytics classifies production incidents into three severity levels. This runbook covers the **escalation path** for SEV-1 (**critical production incidents**) that are not resolved within the initial response window.

| Severity | Definition | Initial response SLA |
|----------|------------|---------------------|
| SEV-1 | Complete service outage or data loss affecting all customers | 5 minutes |
| SEV-2 | Partial outage or significant performance degradation affecting a subset of customers | 15 minutes |
| SEV-3 | Minor issues with no customer-facing impact | 60 minutes |

## Escalation Path for Critical Production Incidents

When a **critical production incident** is declared, the on-call Site Reliability Engineer (SRE) has **30 minutes** to diagnose and remediate the issue before the **escalation path** activates. The **escalation path** for **critical production incidents** not resolved within **30 minutes** of the initial page is as follows:

1. **Tier 1 (0–30 minutes):** Primary on-call SRE investigates and remediates. This is the initial response phase. No formal **escalation path** is activated during this window.

2. **Tier 2 (30–60 minutes):** If the incident is not resolved within **30 minutes**, the **escalation path** triggers automatically via PagerDuty. The incident is escalated to the SRE team lead and the Platform Engineering on-call rotation. A dedicated war room is established in Slack (`#incident-war-room`).

3. **Tier 3 (60–120 minutes):** If still unresolved at the 60-minute mark, the VP of Engineering is paged. The incident commander role is formally assigned. Cross-team resources (Database Engineering, ML Engineering) are pulled into the war room. A communication hold is placed on all non-critical **production deployments**.

4. **Tier 4 (120+ minutes):** At 120 minutes, the CTO is notified. An executive steering call is convened. External communications to affected customers are drafted and sent.

## War Room Protocol

When the **escalation path** reaches Tier 2 (after **30 minutes**), the incident commander must:

- Create a dedicated Slack channel and post the incident timeline.
- Assign roles: communications lead, technical lead, scribe.
- Begin a running incident document with timestamps of all actions.
- Initiate a 15-minute update cadence to all stakeholders.

## Post-Incident Requirements

Following any **critical production incident** that reached Tier 2 of the **escalation path** or higher:

- A 5 Whys analysis must be completed within 48 hours.
- A postmortem document must be published within 5 business days.
- Action items must be tracked in Jira with owners and due dates.
- The **escalation path** itself is reviewed annually to ensure appropriate time windows and contact rosters are current.

## Revision History

This runbook was last updated on 28 June 2026 following a postmortem review of the Q2 indexing service incident.
