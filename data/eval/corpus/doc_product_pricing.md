# Meridian Analytics — Pricing Tier Configuration for Q2 2026 SaaS Product Launch

**Document ID:** doc_product_pricing
**Owner:** Product Management
**Last updated:** 2026-03-20

## Overview

This document describes how pricing tiers should be configured for the new SaaS product launching in Q2 2026 at Meridian Analytics. The new product, Meridian Context Engine, is an AI-powered contextual retrieval service that extends the existing embedding and search platform with dynamic context optimization. This guide covers the pricing tier structure, feature allocation, and configuration parameters for the Q2 2026 launch.

## Pricing tier structure

The pricing tiers for the new SaaS product launching in Q2 2026 should be configured as follows:

**Starter tier — $299 per month**
- 50,000 API requests per month
- 1 GB vector storage
- 384-dimension embeddings only
- 30-day data retention
- Email support (48-hour response SLA)
- Single workspace

**Professional tier — $999 per month**
- 500,000 API requests per month
- 10 GB vector storage
- All dimension sizes (384, 768, 1536, 3072)
- 90-day data retention
- Standard support (8-hour response SLA)
- Up to 5 workspaces
- Custom metadata fields

**Enterprise tier — Custom pricing**
- Unlimited API requests (fair use policy applies)
- 100 GB baseline vector storage, expandable
- All features from Professional tier
- 365-day data retention
- Premium support (1-hour response SLA)
- Unlimited workspaces
- Dedicated infrastructure option
- SSO/SAML integration
- Custom model fine-tuning

## Configuration steps for Q2 2026 launch

To configure the pricing tiers for the new SaaS product launching in Q2 2026, the Product Operations team must follow these steps:

1. **Set up the tier definitions in the billing system:** Create three product SKUs (starter, professional, enterprise) in the billing platform configuration. Each SKU must have the corresponding monthly price, feature access flags, and usage limit parameters.

2. **Configure usage metering:** Define the metering rules for API request counting, vector storage measurement, and data retention period tracking. Usage metering must align with the tier limits specified above.

3. **Set up feature flags:** Configure feature flag toggles in the application backend for each pricing tier. The feature flags control dimension availability, workspace limits, support SLA levels, and SSO access.

4. **Create trial conversion flow:** Configure a 14-day free trial on the Professional tier for all new sign-ups. At trial expiration, users are downgraded to the Starter tier unless a paid plan is selected.

5. **Test tier enforcement:** Run end-to-end tests to verify that usage limits are correctly enforced and that upgrade/downgrade transitions apply the correct feature flags and pricing adjustments.

## Pricing rationale

The pricing tiers are designed to align with the usage patterns observed in the existing Meridian Search Core and Embedding Engine products. The Starter tier targets individual developers and small teams evaluating the platform. The Professional tier serves growing teams with moderate production workloads. The Enterprise tier addresses large organizations with high throughput, compliance, and dedicated infrastructure requirements. The pricing for Q2 2026 includes a 15% launch discount for annual commitments.

## Revision history

This document was last updated on 20 March 2026. Pricing tiers for the Q2 2026 launch are subject to final approval by the Chief Product Officer.
