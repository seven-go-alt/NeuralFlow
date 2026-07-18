# Meridian Analytics — Data Encryption Standards for Document Storage

**Document ID:** doc_tech_encryption
**Owner:** Security Engineering
**Last updated:** 2026-06-20

## Scope

This document defines the **encryption algorithm** standards applied to **data at rest** within Meridian Analytics' document storage service. All customer-uploaded documents, including PDFs, CSVs, and plaintext files, are subject to these encryption policies regardless of storage tier.

## Encryption Algorithm for Data at Rest

The **encryption algorithm** used for all **data at rest** in the document storage service is **AES-256** in Galois/Counter Mode (GCM). This choice satisfies the compliance requirements for SOC 2 Type II and SOC 3 certifications that Meridian maintains.

Key characteristics of the implementation:

- **Algorithm:** **AES-256**-GCM with a 256-bit key derived from the master key using HKDF (RFC 5869) with SHA-256.
- **Initialization vector:** 12-byte random nonce generated per-object using a cryptographically secure random number generator.
- **Additional authenticated data (AAD):** Includes the object ID and the Meridian tenant ID to bind the ciphertext to its context.
- **Key wrapping:** The per-object **AES-256** key is wrapped using the master key with AES-256 Key Wrap (RFC 3394) before being stored alongside the encrypted object metadata.

## Encryption Implementation Details

The document storage service encrypts **data at rest** at the object level before writing to S3-compatible storage. The encryption layer is implemented as a middleware component in the storage pipeline:

```
Upload → TLS termination → Object metadata extraction →
AES-256-GCM encrypt → Key wrap → S3 put
```

Each document object is encrypted with a unique **AES-256** data key. The wrapped data key is stored as object metadata, allowing independent key rotation at the master key level without re-encrypting stored objects.

## Key Management

All master keys used for **data at rest** encryption are managed through Meridian's Hardware Security Module (HSM) cluster. The key hierarchy operates as follows:

1. **Root of trust:** HSM-stored master key, rotated every 12 months.
2. **Storage service key:** Derived from the master key at service startup, cached in memory for the service lifetime.
3. **Per-object data keys:** Generated on upload, used for **AES-256** encryption, wrapped with the storage service key.

The **encryption algorithm** (**AES-256**) and key management scheme were validated by an external security auditor in Q1 2026 with no findings.

## Compliance and Auditing

All encryption operations are logged with the following details:

- Object ID and tenant ID
- Timestamp of encryption and decryption events
- Key ID (wrapping key identifier)
- Authentication result (decryption only)

Logs are securely forwarded to Meridian's SIEM platform and retained for 7 years to meet regulatory requirements. Decryption events are additionally flagged for review when they originate from non-production IP ranges.

## Revision History

This document was last updated on 20 June 2026 following the Q2 key rotation and external audit validation.
