"""
saas/orgs.py
First-time signup flow: turn a newly authenticated user into an org owner.
"""
from __future__ import annotations

from sqlalchemy.engine import Engine

from .db import add_membership, create_organization, get_org_for_user


def bootstrap_org(engine: Engine, user_id: str, org_name: str) -> str:
    """
    Create a new organization owned by user_id, or return their existing org
    if they already belong to one (idempotent — safe to call on every login).
    """
    existing = get_org_for_user(engine, user_id)
    if existing:
        return existing

    org_id = create_organization(engine, name=org_name)
    add_membership(engine, user_id=user_id, org_id=org_id, role="owner")
    return org_id
