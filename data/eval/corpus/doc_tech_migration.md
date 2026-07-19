# Meridian Analytics — Alembic Database Migration Rollback with Column Rename

**Document ID:** doc_tech_migration
**Owner:** Platform Data Team
**Last updated:** 2026-07-05

## Overview

Meridian Analytics manages schema changes across dozens of PostgreSQL databases using **Alembic** as the **database migration** framework. Columns are frequently renamed to match evolving business terminology. A column rename presents a unique challenge for migration **roll back**: the **downgrade** function must reverse the rename without data loss. This document describes the correct method for rolling back an Alembic migration when a **column rename** was performed in the upgrade.

## The Column Rename Migration Structure

When a column is renamed, the upgrade path uses Alembic's `alter_column` operation. The following example renames `account_manager` to `relationship_manager` in the `customer_accounts` table:

```python
"""Rename account_manager to relationship_manager

Revision ID: 8a5f3e2d1c0b
Revises: 7b4e3d2c1a0f
"""

from alembic import op
import sqlalchemy as sa

revision = "8a5f3e2d1c0b"
down_revision = "7b4e3d2c1a0f"

def upgrade():
    op.alter_column(
        "customer_accounts", "account_manager",
        new_column_name="relationship_manager",
        existing_type=sa.String(255),
        existing_nullable=True,
    )

def downgrade():
    op.alter_column(
        "customer_accounts", "relationship_manager",
        new_column_name="account_manager",
        existing_type=sa.String(255),
        existing_nullable=True,
    )
```

The downgrade above may fail if views or stored procedures reference the renamed column. The correct method must account for these dependencies.

## Correct Downgrade Method with Dependency Handling

Rolling back a column rename requires three steps in the downgrade function:

1. **Drop dependent objects**: Drop any views or functions referencing the new name.
2. **Reverse the column rename**: Use `op.alter_column` with the new name as source.
3. **Restore dependent objects**: Recreate them with the old column name.

```python
def downgrade():
    # Step 1: Drop dependent views
    op.execute("DROP VIEW IF EXISTS customer_accounts_summary CASCADE")

    # Step 2: Reverse the column rename
    op.alter_column(
        "customer_accounts", "relationship_manager",
        new_column_name="account_manager",
        existing_type=sa.String(255),
        existing_nullable=True,
    )

    # Step 3: Recreate views with old column name
    op.execute("""
        CREATE VIEW customer_accounts_summary AS
        SELECT id, customer_name, account_manager, account_tier, created_at
        FROM customer_accounts WHERE status = 'active'
    """)
```

## Data Integrity Considerations During Rollback

The column rename in PostgreSQL acquires an `ACCESS EXCLUSIVE` lock on the table. For a metadata-only rename, lock duration is sub-second, but dropping and recreating dependent objects may extend the window. Meridian requires rollback migrations be applied during a maintenance window with the application in read-only mode. The staging validation must verify:

- All data in the renamed column is preserved after downgrade.
- All dependent objects compile correctly after the rollback.
- Foreign key constraints referencing the column remain valid.

## Preventing Data Loss During Downgrade

The most critical risk during column rename rollback is data truncation. If source and target column types differ, `alter_column` may silently truncate data. Meridian enforces a policy that column rename migrations must use the same column type on both sides. If a type change is required, it must be performed as a separate migration step with its own downgrade logic.

Every column rename migration must include a data integrity assertion in the downgrade. Before executing the rename reversal, the downgrade queries the new column for non-null values and logs a warning if all values are null, indicating the old column may have already been dropped or renamed elsewhere. This early detection prevents silent data loss and gives the operator a clear signal before the migration proceeds.
